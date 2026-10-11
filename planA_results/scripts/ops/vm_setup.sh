#!/bin/bash
# Prepares a Debian 13 GCP VM for the Plan A pipeline. Run as root, once per VM:
#   gcloud compute ssh <vm> --zone <zone> -- 'sudo bash -s' < vm_setup.sh
#
# 1. user + passwordless sudo + Claude's SSH public key
# 2. system packages for A1/A2/A4/A5/A6 (git/patch for applying patches, python3-venv for
#    per-task venvs, build-essential for test deps without wheels, numpy/scipy, rsync/pigz/tmux)
# 3. ~/a3env: Python venv for A3 (encoder audit) and A5 mode (c): sentence-transformers,
#    CPU-only torch + torchvision, numpy, scipy, pillow (pillow and torchvision are required by
#    EmbeddingGemma-2's multimodal processor)
# 4. Hugging Face access (manual, see the message printed at the end): EmbeddingGemma-300m is a
#    gated model, so A3 needs a read token saved in ~/.hf_token on the VM.
set -e
U=sauravbellam
KEY='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJJrS/T8gGVElIwD8CFPxeiJtwwxAblfCDzTqmLl7RSl claude-graphcomp'

# --- 1. user, sudo, SSH key
id "$U" >/dev/null 2>&1 || useradd -m -s /bin/bash "$U"
echo "$U ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-claude && chmod 440 /etc/sudoers.d/90-claude
install -d -m 700 -o "$U" -g "$U" /home/$U/.ssh
touch /home/$U/.ssh/authorized_keys
grep -q claude-graphcomp /home/$U/.ssh/authorized_keys || echo "$KEY" >> /home/$U/.ssh/authorized_keys
chown "$U":"$U" /home/$U/.ssh/authorized_keys && chmod 600 /home/$U/.ssh/authorized_keys

# --- 2. system packages
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq rsync git patch pigz build-essential python3-venv python3-pip python3-numpy python3-scipy tmux >/dev/null

# --- 3. A3 / A5(c) environment (as the user, in ~/a3env)
sudo -u "$U" -H bash -c '
  [ -x ~/a3env/bin/python ] || python3 -m venv ~/a3env
  ~/a3env/bin/pip install -q --disable-pip-version-check sentence-transformers numpy scipy pillow torch torchvision \
      --extra-index-url https://download.pytorch.org/whl/cpu
  ~/a3env/bin/python -c "import sentence_transformers, torch, torchvision, PIL; print(\"a3env ok: sentence-transformers\", sentence_transformers.__version__, \"torch\", torch.__version__)"
'

echo "DONE on $(hostname): $(python3 --version), sudo ok for $U, ~/a3env ready"
cat <<'MSG'

--- Hugging Face access (needed for A3 only; do this once, by hand) -----------------------------
 1. On huggingface.co, open google/embeddinggemma-300m and accept the licence.
 2. Settings -> Access Tokens -> create a READ token.
 3. In an SSH shell on this VM (as the user, not root), save it without echoing it:
      install -m 600 /dev/null ~/.hf_token && read -rs -p "HF token: " t && echo "$t" > ~/.hf_token && unset t && echo "saved $(wc -c < ~/.hf_token) bytes"
    It should report about 38 bytes. Never paste the token into a chat.
 4. A3 then runs with:  export HF_TOKEN=$(tr -d "[:space:]" < ~/.hf_token)
-------------------------------------------------------------------------------------------------
MSG
