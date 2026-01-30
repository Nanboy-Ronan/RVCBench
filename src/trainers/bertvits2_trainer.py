# gan_hf_trainer.py
import os, torch
from src.utils.commons import set_requires_grad
from torch.optim.lr_scheduler import ExponentialLR
from transformers import Trainer, TrainerCallback, TrainerState, TrainerControl

class _EpochLrStepCallback(TrainerCallback):
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        trainer: "BertVITS2Trainer" = kwargs["trainer"]
        # Step every scheduler in sync
        if trainer.lr_scheduler_g  is not None: trainer.lr_scheduler_g.step()
        if trainer.lr_scheduler_d  is not None: trainer.lr_scheduler_d.step()
        if trainer.lr_scheduler_wd is not None: trainer.lr_scheduler_wd.step()
        if trainer.lr_scheduler_dur is not None: trainer.lr_scheduler_dur.step()
        return control

class BertVITS2Trainer(Trainer):
    def __init__(self, *args, epoch_start: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_start = int(epoch_start)  # Used to compute last_epoch
        # Placeholders
        self.optimizer_g = self.optimizer_d = self.optimizer_wd = self.optimizer_dur = None
        self.lr_scheduler_g = self.lr_scheduler_d = self.lr_scheduler_wd = self.lr_scheduler_dur = None

        # Attach the per-epoch scheduler callback (see section 2 below)
        self.add_callback(_EpochLrStepCallback())

    def create_optimizer_and_scheduler(self, num_training_steps):
        # ---------- A) Optimizers (kept from original implementation) ----------
        self.optimizer_g = torch.optim.AdamW(self.model.net_g.parameters(),
                                             lr=self.args.learning_rate, betas=(0.9, 0.999))
        self.optimizer_d = torch.optim.AdamW(self.model.net_d.parameters(),
                                             lr=self.args.learning_rate, betas=(0.9, 0.999))
        self.optimizer_wd = torch.optim.AdamW(self.model.net_wd.parameters(),
                                              lr=self.args.learning_rate, betas=(0.9, 0.999))
        self.optimizer_dur = None
        if getattr(self.model, "net_dur_disc", None) is not None:
            self.optimizer_dur = torch.optim.AdamW(self.model.net_dur_disc.parameters(),
                                                   lr=self.args.learning_rate, betas=(0.9, 0.999))

        # Wrap with accelerate (DDP/AMP)
        prepared = self.accelerator.prepare(
            self.optimizer_g,
            self.optimizer_d,
            self.optimizer_wd,
            *([self.optimizer_dur] if self.optimizer_dur is not None else [])
        )
        self.optimizer_g, self.optimizer_d, self.optimizer_wd = prepared[:3]
        if self.optimizer_dur is not None and len(prepared) == 4:
            self.optimizer_dur = prepared[3]

        # ---------- B) Schedulers (inlined from build_schedulers) ----------
        gamma = float(self.args.weight_decay)  # Keep consistent with build_schedulers

        # Match the original last_epoch=epoch_str-2 behavior
        # epoch_str is the epoch index that training resumes from; reuse epoch_start from __init__
        epoch_str = int(self.epoch_start)
        last_epoch = epoch_str - 2

        self.lr_scheduler_g = ExponentialLR(self.optimizer_g, gamma=gamma, last_epoch=last_epoch)
        self.lr_scheduler_d = ExponentialLR(self.optimizer_d, gamma=gamma, last_epoch=last_epoch)
        self.lr_scheduler_wd = ExponentialLR(self.optimizer_wd, gamma=gamma, last_epoch=last_epoch)
        self.lr_scheduler_dur = (ExponentialLR(self.optimizer_dur, gamma=gamma, last_epoch=last_epoch)
                                 if self.optimizer_dur is not None else None)

        # ---------- C) Expose main optimizer/scheduler to the parent class ----------
        self.optimizer = self.optimizer_g
        self.lr_scheduler = self.lr_scheduler_g

    def save_optimizer_and_scheduler(self, output_dir: str):
        super().save_optimizer_and_scheduler(output_dir)  # Save generator state
        torch.save(self.accelerator.get_state_dict(self.optimizer_d), os.path.join(output_dir, "optimizer_d.pt"))
        torch.save(self.accelerator.get_state_dict(self.optimizer_wd), os.path.join(output_dir, "optimizer_wd.pt"))
        if self.optimizer_dur is not None:
            torch.save(self.accelerator.get_state_dict(self.optimizer_dur),
                       os.path.join(output_dir, "optimizer_dur.pt"))
        # Persist each scheduler
        torch.save(self.lr_scheduler_d.state_dict(), os.path.join(output_dir, "scheduler_d.pt"))
        torch.save(self.lr_scheduler_wd.state_dict(), os.path.join(output_dir, "scheduler_wd.pt"))
        if self.lr_scheduler_dur is not None:
            torch.save(self.lr_scheduler_dur.state_dict(), os.path.join(output_dir, "scheduler_dur.pt"))

    def _load_optimizer_and_scheduler(self, checkpoint: str):
        if not checkpoint:
            return
        super()._load_optimizer_and_scheduler(checkpoint)  # Restore generator state
        p = lambda n: os.path.join(checkpoint, n)
        if os.path.exists(p("optimizer_d.pt")):
            self.optimizer_d.load_state_dict(torch.load(p("optimizer_d.pt"), map_location="cpu"))
        if os.path.exists(p("optimizer_wd.pt")):
            self.optimizer_wd.load_state_dict(torch.load(p("optimizer_wd.pt"), map_location="cpu"))
        if self.optimizer_dur is not None and os.path.exists(p("optimizer_dur.pt")):
            self.optimizer_dur.load_state_dict(torch.load(p("optimizer_dur.pt"), map_location="cpu"))
        if os.path.exists(p("scheduler_d.pt")):
            self.lr_scheduler_d.load_state_dict(torch.load(p("scheduler_d.pt"), map_location="cpu"))
        if os.path.exists(p("scheduler_wd.pt")):
            self.lr_scheduler_wd.load_state_dict(torch.load(p("scheduler_wd.pt"), map_location="cpu"))
        if self.lr_scheduler_dur is not None and os.path.exists(p("scheduler_dur.pt")):
            self.lr_scheduler_dur.load_state_dict(torch.load(p("scheduler_dur.pt"), map_location="cpu"))

    def _select_phase(self, inputs):
        # Prefer an explicit phase/step from the batch; otherwise rotate by step count
        phase = inputs.pop("phase", None) or inputs.pop("step", None)
        if phase is not None:
            return phase

        # Rotate by global_step: D -> WD -> DUR -> G (run stages only if available)
        i = int(self.state.global_step) % 4
        order = ["D", "WD", "DUR", "G"]
        # Skip stages without the corresponding module
        if not hasattr(self.model, "net_d"):        order.remove("D")
        if not hasattr(self.model, "wl") or self.model.wl is None: order.remove("WD")
        if not hasattr(self.model, "net_dur_disc") or self.model.net_dur_disc is None: order.remove("DUR")
        if "G" not in order: order.append("G")
        return order[i % len(order)]

    def _toggle_modules(self, phase: str):
        # Enable gradients only for the submodule used in this phase
        # G
        set_requires_grad(getattr(self.model, "net_g", None), phase == "G")
        # Discriminator D
        set_requires_grad(getattr(self.model, "net_d", None), phase == "D")
        # SLM discriminator (WavLMDiscriminator / WavLMLoss may have trainable parameters)
        set_requires_grad(getattr(self.model, "net_wd", None), phase == "WD")
        # Some implementations store discriminator weights in self.wl; unfreeze if present
        set_requires_grad(getattr(self.model, "wl", None), phase == "WD")
        # Duration discriminator
        set_requires_grad(getattr(self.model, "net_dur_disc", None), phase == "DUR")

    def training_step(self, model, inputs, num_items_in_batch, **kwargs):
        # Optional: sync MAS noise scaling
        if hasattr(model, "set_train_step"):
            model.set_train_step(self.state.global_step)

        amp_ctx = self.autocast_smart_context_manager()

        inputs= {'batch_data':inputs}
        # ------- 1) D -------
        self.optimizer_d.zero_grad(set_to_none=True)
        with amp_ctx:
            loss_d, _ = self.compute_loss(
                model,
                inputs={**inputs, "phase": "D"},
                return_outputs=True,
                num_items_in_batch=num_items_in_batch
            )
        loss_d.backward()
        self.optimizer_d.step()
        if self.lr_scheduler_d is not None:
            self.lr_scheduler_d.step()

        # ------- 2) DUR (optional) -------
        if self.optimizer_dur is not None:
            self.optimizer_dur.zero_grad(set_to_none=True)
            with amp_ctx:
                loss_dur, _ = self.compute_loss(
                    model,
                    inputs={**inputs, "phase": "DUR"},
                    return_outputs=True,
                    num_items_in_batch=num_items_in_batch
                )
            loss_dur.backward()
            self.optimizer_dur.step()
            if self.lr_scheduler_dur is not None:
                self.lr_scheduler_dur.step()
        else:
            loss_dur = 0.0

        # ------- 3) WD (if wl/net_wd is available) -------
        if (hasattr(self.model, "wl") and self.model.wl is not None) or hasattr(self.model, "net_wd"):
            self.optimizer_wd.zero_grad(set_to_none=True)
            with amp_ctx:
                loss_wd, _ = self.compute_loss(
                    model,
                    inputs={**inputs, "phase": "WD"},
                    return_outputs=True,
                    num_items_in_batch=num_items_in_batch
                )
            loss_wd.backward()
            self.optimizer_wd.step()
            if self.lr_scheduler_wd is not None:
                self.lr_scheduler_wd.step()
        else:
            loss_wd = 0.0

        # ------- 4) G -------
        self.optimizer_g.zero_grad(set_to_none=True)
        with amp_ctx:
            loss_g, _ = self.compute_loss(
                model,
                inputs={**inputs, "phase": "G"},
                return_outputs=True,
                num_items_in_batch=num_items_in_batch
            )
        loss_g.backward()
        self.optimizer_g.step()
        if self.lr_scheduler_g is not None:
            self.lr_scheduler_g.step()

        # Aggregate losses for logging
        loss_gen_all = loss_g + loss_d + loss_dur + loss_wd

        return loss_gen_all

    def compute_loss(self, model, inputs,return_outputs=False, num_items_in_batch=None):
        '''Note: num_items_in_batch is not considered for maintainance'''
        # Optional: push step information into the generator for internal schedules
        if hasattr(model, "set_train_step"):
            model.set_train_step(int(self.state.global_step))

        phase = self._select_phase(inputs)
        self._toggle_modules(phase)

        # Call the model forward(step=...)
        out = model(step=phase, **inputs)  # -> GANStepOutput(loss=..., extras={...}) todo fix ugly code
        loss = out.loss
        # ========= Key change: respect num_items_in_batch =========
        if num_items_in_batch is not None:
            # Assume the loss is a sum and normalize accordingly
            if loss.dim() > 0:  # Some models may return per-example loss
                loss = loss.mean()
            else:
                loss = loss / num_items_in_batch
        # Log metrics from extras (displayed in progress bar or wandb callbacks)
        if getattr(out, "extras", None):
            log_dict = {}
            for k, v in out.extras.items():
                try:
                    val = v.detach().item() if torch.is_tensor(v) else float(v)
                    if torch.isfinite(torch.tensor(val)):
                        log_dict[k] = val
                except Exception:
                    pass
            if log_dict:
                # Include the current phase for clarity
                log_dict["phase"] = {"D":0, "WD":1, "DUR":2, "G":3}.get(phase, 99)
                self.log(log_dict)

        return (loss, out) if return_outputs else loss
