
SESSION_NAME="palette"
tasks="translateall translatebone inpaintbone"
ports=(21012 21013 21014)
# Start new tmux session
tmux new-session -d -s $SESSION_NAME

ndx=0
for task in ${tasks}; do 
name=0216_"$task"
COMMAND="""cd /home/yua4/repos/Palette-Image-to-Image-Diffusion-Models
source /home/yua4/ptoa/venv/bin/activate
python run.py \
-c config/240216_$task.json \
-P ${ports[$ndx]}
"""
echo $COMMAND
tmux new-window -t $SESSION_NAME -n "$name"
tmux send-keys -t "${SESSION_NAME}:${name}" "$COMMAND" C-m
((ndx++))
done

# Attach to the new session
tmux attach -t $SESSION_NAME
