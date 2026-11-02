import os
import re
import signal
import sys
from pathlib import Path

import torch
import logging
from tqdm import tqdm

# [FEAT-TSWEEP-2026-10-16] AMP helpers for the temperature rescue sweep: prefer the
# modern torch.amp API and fall back to torch.cuda.amp on older torch builds, so this
# file imports on any 2.x server. Inert unless args.amp is set.
try:
    torch.amp.autocast, torch.amp.GradScaler

    def _autocast(enabled):
        return torch.amp.autocast(device_type='cuda', enabled=enabled)

    def _grad_scaler(enabled):
        return torch.amp.GradScaler('cuda', enabled=enabled)
except AttributeError:
    def _autocast(enabled):
        return torch.cuda.amp.autocast(enabled=enabled)

    def _grad_scaler(enabled):
        return torch.cuda.amp.GradScaler(enabled=enabled)


# [FEAT-RESUME-2026-10-16] Checkpoint-resume support (server-coexistence program).
# last.pt lives next to args.checkpoints: <dir>/<stem>-last.pt and stores model /
# optimizer / scheduler / scaler states plus the number of COMPLETED epochs.
# - refreshed at every epoch end;
# - on SIGTERM/SIGINT the training loop saves immediately (worst-case progress loss
#   = the epoch in progress) and exits with code 2, which mam_queue records as
#   'interrupted' (relaunchable) rather than 'failed';
# - a second signal restores the default handler and kills at once (the previous
#   epoch's last.pt still holds).
# Off unless the entry was launched with --resume auto (getattr-guarded, so entries
# without the flag keep the exact original behavior).
def _last_pt_path(args):
    ckpt = Path(args.checkpoints)
    return ckpt.parent / (ckpt.stem + '-last.pt')


def _save_last(args, model, optimizer, scheduler, scaler, completed, best):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'epochs_completed': completed,
             'best_loss': best['loss'], 'best_epoch': best['epoch']}
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
    if scaler is not None:
        state['scaler'] = scaler.state_dict()
    tmp = str(_last_pt_path(args)) + '.writing'
    torch.save(state, tmp)
    os.replace(tmp, _last_pt_path(args))


_BEST_RE = re.compile(
    r'New best model found at epoch (\d+) with Eval Loss: (\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)')
# [FIX-CAUGHT-BY-TEST-2026-10-17] the earlier char class [0-9.eE+-] also matched
# the sentence's trailing '.', so float('3.5000000.') crashed the resume path.


def _best_from_log(log_path):
    """Last best-val record of this run's own log; used when resuming from a
    last.pt written before best_loss/best_epoch were persisted."""
    try:
        text = Path(log_path).read_text(errors='replace')
    except OSError:
        return None
    hits = _BEST_RE.findall(text)
    if not hits:
        return None
    try:
        return {'epoch': int(hits[-1][0]), 'loss': float(hits[-1][1])}
    except ValueError:
        # a malformed record must never crash a resume; fall back to the
        # explicit RESUME:::WARNING path instead
        return None


def _patience(args):
    """Early-stop patience in epochs; 0 = off. --early_stop_patience wins, else a
    '<checkpoint stem>.early_stop' sidecar holding an int (re-read every epoch so a
    running job can be switched without changing its launch args)."""
    p = int(getattr(args, 'early_stop_patience', 0) or 0)
    if p > 0:
        return p
    side = Path(args.checkpoints).with_suffix('.early_stop')
    try:
        return max(int(side.read_text().strip()), 0)
    except (OSError, ValueError):
        return 0


def _install_stop_handler(stop):
    def handler(signum, frame):
        if stop['signal'] is not None:
            # second signal: the user wants the GPU back NOW; die at once and
            # keep the last completed-epoch checkpoint on disk
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        stop['signal'] = signum
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass  # not the main thread: per-iteration flag checks still apply


def setup_logging(log_path):
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(message)s')

def evaluate_model(model, eval_loader, device):
    model.eval()

    mse_loss_fn = torch.nn.MSELoss()
    mae_loss_fn = torch.nn.L1Loss()
    total_mse_loss = 0.0
    total_mae_loss = 0.0
    total_mape_loss = 0.0

    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(eval_loader,desc="Evaluating model", leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            mse_loss = mse_loss_fn(outputs, targets)
            total_mse_loss += mse_loss.item() * inputs.size(0)

            mae_loss = mae_loss_fn(outputs, targets)
            total_mae_loss += mae_loss.item() * inputs.size(0)

            total_samples += inputs.size(0)

    avg_eval_mse_loss = total_mse_loss / total_samples
    avg_eval_mae_loss = total_mae_loss / total_samples

    logging.info(f'EVAL:::MSE: {avg_eval_mse_loss:.7f}, MAE: {avg_eval_mae_loss:.7f}')
    return avg_eval_mse_loss, avg_eval_mae_loss

def test_model(model, test_loader, device):
    model.eval()

    mse_loss_fn = torch.nn.MSELoss()
    mae_loss_fn = torch.nn.L1Loss()
    total_mse_loss = 0.0
    total_mae_loss = 0.0

    total_samples = 0
    with torch.no_grad():
        progress_bar = tqdm(test_loader,desc="Testing model", leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            mse_loss = mse_loss_fn(outputs, targets)
            total_mse_loss += mse_loss.item() * inputs.size(0)

            mae_loss = mae_loss_fn(outputs, targets)
            total_mae_loss += mae_loss.item() * inputs.size(0)

            total_samples += inputs.size(0)
    avg_test_mse_loss = total_mse_loss / total_samples
    avg_test_mae_loss = total_mae_loss / total_samples
    logging.info(f'TEST:::MSE: {avg_test_mse_loss:.7f}, MAE: {avg_test_mae_loss:.7f}')
    return avg_test_mse_loss, avg_test_mae_loss


def train_model(model, train_loader, eval_loader, criterion, optimizer, device, args):
    logs_path = args.log_path
    setup_logging(logs_path)

    # [FEAT-TSWEEP-2026-10-16] Optional cosine LR decay + CUDA AMP for the temperature
    # rescue sweep. Both default OFF via getattr, so entries without these argparse
    # flags (and every existing run) behave exactly as before.
    lr_schedule = getattr(args, 'lr_schedule', 'const')
    use_amp = bool(getattr(args, 'amp', False))
    # [FEAT-RESUME-2026-10-16] 'auto' = continue from last.pt if present
    resume = getattr(args, 'resume', 'no')
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
                 if lr_schedule == 'cosine' else None)
    scaler = _grad_scaler(use_amp)

    if args.if_pretrained == True:
        model.load_state_dict(torch.load(args.checkpoints))
        print(f'Loaded model from {args.checkpoints}')

    # [FEAT-RESUME-2026-10-16] restore state; scheduler/scaler only when active
    start_epoch = 0
    best = {'loss': float('inf'), 'epoch': 0}
    last_path = _last_pt_path(args)
    if resume == 'auto':
        if Path(last_path).exists():
            # [FIX-GPU-RESUME-2026-10-17] load to CPU. Loading with
            # map_location=<cuda device> put Adam state_steps on the GPU and the
            # FIRST optimizer step asserted ("If capturable=False, state_steps
            # should not be CUDA tensors") on this torch build — caught live when
            # corr_mm_seed2_v2 was resumed on GPU2 (three relaunch attempts
            # burned in ~30s each; CPU-only regression tests cannot see it).
            state = torch.load(str(last_path), map_location='cpu')
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            # Normalize placement version-independently: torch's load_state_dict
            # device-casting differs across builds, and Adam requires state_steps
            # on CPU (capturable=False) with moment buffers on the param device.
            for group in optimizer.param_groups:
                for p in group['params']:
                    st = optimizer.state.get(p)
                    if not st:
                        continue
                    for k, v in st.items():
                        if not torch.is_tensor(v):
                            continue
                        if k == 'step':
                            if v.device.type != 'cpu':
                                st[k] = v.cpu()
                        elif v.device != p.device:
                            st[k] = v.to(p.device)
            start_epoch = int(state.get('epochs_completed', 0))
            # [FIX-RESUME-BEST-2026-10-17] without this, best_loss restarted at inf
            # on every resume and the first post-resume epoch overwrote the
            # best-val checkpoint even when it was worse.
            if state.get('best_loss') is not None:
                best = {'loss': float(state['best_loss']),
                        'epoch': int(state.get('best_epoch', 0))}
            else:
                from_log = _best_from_log(logs_path)
                if from_log is not None:
                    best = from_log
                else:
                    print('RESUME:::WARNING best-val record not found; best checkpoint '
                          'may be overwritten by the next epoch')
            logging.info(f'RESUME:::best so far epoch {best["epoch"]} '
                         f'val {best["loss"]:.7f}')
            if scheduler is not None and state.get('scheduler') is not None:
                scheduler.load_state_dict(state['scheduler'])
            if state.get('scaler') is not None:
                scaler.load_state_dict(state['scaler'])
            print(f'RESUME:::continuing from epoch {start_epoch + 1} via {last_path}')
            logging.info(f'RESUME:::continuing from epoch {start_epoch + 1} via {last_path}')
        else:
            print(f'RESUME:::no last.pt at {last_path}, starting fresh')
            logging.info(f'RESUME:::no last.pt at {last_path}, starting fresh')

    stop = {'signal': None}
    if resume == 'auto':
        _install_stop_handler(stop)

    def _early_stop_due(completed):
        patience = _patience(args)
        if patience > 0 and best['epoch'] > 0 and completed - best['epoch'] >= patience:
            msg = (f'EARLY_STOP:::after epoch {completed}: no val improvement for '
                   f'{completed - best["epoch"]} epochs (patience {patience}); best epoch '
                   f'{best["epoch"]} val {best["loss"]:.7f}')
            logging.info(msg)
            print(msg)
            return True
        return False

    if _early_stop_due(start_epoch):
        print("Training complete.")
        return

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        count = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.num_epochs}', leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with _autocast(use_amp):
                preds = model(inputs)
                loss = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            progress_bar.set_postfix(loss=loss.item())

            count += 1
            if count % args.log_per_samples == 0:
                logging.info(f'TRAIN:::sample {count + 1} - Train Loss: {loss.item():.4f}')

            if stop['signal'] is not None:
                _save_last(args, model, optimizer, scheduler, scaler, epoch, best)
                logging.info(f'TRAINING INTERRUPTED:::during epoch {epoch + 1}; '
                             f'last.pt saved with {epoch} completed epochs')
                print('TRAINING INTERRUPTED:::last.pt saved')
                raise SystemExit(2)

        average_loss = total_loss / total_samples
        logging.info(f'TRAIN FINISHED:::Epoch {epoch + 1}, Train MSE Loss: {average_loss:.7f}')
        print(f'TRAIN FINISHED:::Epoch {epoch + 1}, Train MSE Loss: {average_loss:.7f}')
        eval_mse_loss, eval_mae_loss = evaluate_model(model, eval_loader, device)
        print(f'EVAL:::MSE Loss: {eval_mse_loss:.7f}, MAE Loss: {eval_mae_loss:.7f}')

        if eval_mse_loss < best['loss']:
            best = {'loss': eval_mse_loss, 'epoch': epoch + 1}
            logging.info(f'New best model found at epoch {epoch + 1} with Eval Loss: {eval_mse_loss:.7f}.')
            print(f'New best model found at epoch {epoch + 1} with loss {eval_mse_loss:.7f}. Saving model...')
            torch.save(model.state_dict(), args.checkpoints)

        # [FEAT-TSWEEP-2026-10-16] per-epoch LR decay step (sweep runs only)
        if scheduler is not None:
            scheduler.step()
            logging.info(f'SCHEDULE:::lr {optimizer.param_groups[0]["lr"]:.6e}')

        # [FEAT-RESUME-2026-10-16] refresh the resumable checkpoint every epoch
        _save_last(args, model, optimizer, scheduler, scaler, epoch + 1, best)
        if stop['signal'] is not None:
            logging.info(f'TRAINING INTERRUPTED:::after epoch {epoch + 1}; last.pt saved')
            print('TRAINING INTERRUPTED:::last.pt saved')
            raise SystemExit(2)
        if _early_stop_due(epoch + 1):
            break

    print("Training complete.")
