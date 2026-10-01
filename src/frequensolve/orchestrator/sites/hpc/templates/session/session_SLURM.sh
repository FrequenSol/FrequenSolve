#!/bin/bash
#SBATCH -J {{ name }}
#SBATCH -o {{ log_path }}
#SBATCH -e {{ log_path }}
#SBATCH -N {{ nodes }}
#SBATCH -n {{ ranks }}
#SBATCH --ntasks-per-node={{ ranks_per_node }}
{% if cpus_per_task %}
#SBATCH --cpus-per-task={{ cpus_per_task }}
{% endif %}
#SBATCH -p {{ queue }}
{% if account %}
#SBATCH -A {{ account }}
{% endif %}
#SBATCH -t {{ duration }}
#SBATCH --signal=B:USR1@120

set -euo pipefail
{% for line in runtime_setup %}
{{ line }}
{% endfor %}
# exec allows the controller to receive Slurm's walltime and cancellation signals.
exec python3 {{ root }}/session_controller.py --root {{ root }}
