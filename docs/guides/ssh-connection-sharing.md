# Share one SSH login across FrequenSolve sessions

Many HPC sites require MFA for a new SSH authentication, but they do not require
a new authentication for every remote command. OpenSSH connection sharing keeps
one authenticated master connection available for later SSH and `rsync`
clients.

With connection sharing enabled, one SSH login can be reused by:

- separate Python scripts and notebooks;
- repeated `fs.Site()` construction;
- project and job uploads;
- result downloads; and
- direct `rsync` operations started by FrequenSolve.

No password or MFA token is stored in `site.toml` or in the control socket.

## Recommended: let FrequenSolve manage the connection

After activating the Python environment, run this once near the start of a
work session:

```bash
frequensolve site connect
```

Respond to any password, MFA, or other SSH prompts from the configured site. If
an SSH agent or a normally discovered key can authenticate, OpenSSH uses it
automatically.

The command:

1. reads the username and hostname from `~/.frequensolve/site.toml`;
2. checks for an existing FrequenSolve-managed connection;
3. creates a user-only socket below `~/.ssh/control/` when needed; and
4. leaves the authenticated SSH master running in the background for up to
   eight hours.

It does **not** modify `~/.ssh/config`. Running it again is idempotent: when the
shared connection is still active, the command reports that it is already
available and does not request another authentication.

After it connects, run as many separate scripts or notebooks as needed:

```bash
python first_experiment.py
python second_experiment.py
python -m jupyter lab
```

Each process discovers the same socket. FrequenSolve also passes that exact
socket to `rsync`, so uploads and downloads do not create independent SSH
logins.

Re-run `frequensolve site connect` after the eight-hour connection expires or
after the computer sleeps, restarts, changes networks, or loses access to the
remote host.

## Optional: configure OpenSSH manually

Users who want ordinary `ssh` commands to create reusable connections can opt
into a standard OpenSSH configuration. FrequenSolve never installs this block
automatically.

Create the socket directory and set restrictive permissions:

```bash
mkdir -p "$HOME/.ssh/control"
chmod 700 "$HOME/.ssh" "$HOME/.ssh/control"
touch "$HOME/.ssh/config"
chmod 600 "$HOME/.ssh/config"
```

Add this block to `~/.ssh/config`:

```sshconfig
Host login.example.edu
  ControlMaster auto
  ControlPersist 8h
  ControlPath ~/.ssh/control/%C
  ServerAliveInterval 30
  ServerAliveCountMax 3
```

Replace `login.example.edu` with the `hostname` from the selected FrequenSolve
site profile.

Do not add `IdentityFile` or `IdentitiesOnly` unless the site actually requires
a nonstandard local key. Many sites support password-plus-MFA or SSH-agent
authentication without an explicit key path in `site.toml`.

Open the master connection once:

```bash
ssh YOUR_USERNAME@login.example.edu
```

Use the same username and hostname configured in the selected FrequenSolve
profile.

After authentication succeeds, exit the interactive shell. `ControlPersist`
keeps the master available in the background. Because its socket is under
`~/.ssh/control/`, FrequenSolve discovers it when a later `fs.Site()` is
created and uses it for transfers as well.

This manual configuration and `frequensolve site connect` are alternatives.
The helper uses its own explicitly named socket, while the OpenSSH block lets
normal `ssh` commands create the shared socket.

## Check whether sharing works

For the recommended FrequenSolve-managed connection, run:

```bash
frequensolve site connect
```

An active connection produces an `already available` message without another
authentication prompt. Then verify the complete FrequenSolve site
configuration:

```bash
frequensolve site check
```

For the manual OpenSSH configuration, this command checks the master selected
by `~/.ssh/config`:

```bash
ssh -O check YOUR_USERNAME@login.example.edu
```

## Troubleshooting

### Every script requests authentication

Make sure the initial connection was created from the same local account and
environment used to run FrequenSolve. Run `frequensolve site connect` again;
it should report either a successful new connection or that one is already
available.

For manual configuration, confirm that `ControlPath` points inside
`~/.ssh/control/`. FrequenSolve looks in that directory for reusable OpenSSH
sockets.

### Transfers request authentication

Install local `rsync` and establish the shared connection before constructing
`fs.Site()`. Once a site has discovered the control socket, FrequenSolve passes
the socket explicitly to each `rsync` subprocess.

### The connection stopped working

Laptop sleep, network changes, idle expiration, and restarts can end the
background master. Re-run `frequensolve site connect`; stale
FrequenSolve-managed sockets are checked and replaced automatically.

### OpenSSH reports a changed host key

Do not bypass host-key checking. Compare the reported host and fingerprint with
the HPC site's current guidance before changing `~/.ssh/known_hosts`.

## Security notes

- The socket is local and the containing directory should remain mode `0700`.
- Passwords, private-key contents, passphrases, and MFA tokens do not belong in
  `site.toml` or `~/.ssh/config`.
- Connection sharing grants local processes running as your OS user access to
  the authenticated connection until it expires. Lock your workstation and do
  not use this setup from a shared local login.
- The HPC site's login, host-key, and MFA policies remain authoritative. Follow
  its current documentation and security requirements.
