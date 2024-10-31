import subprocess

# Define the list of tasks
tasks = ["foveal_scan", "healthy", "drusen", "ped", "hdots"]
# model_name
# Define the command template
command_template = "python task_specific_testing.py --model_name=retfound --device_id=3 --batch_size=8 --task={}"

# Iterate over each task and run the command
for task in tasks:
    command = command_template.format(task)
    print("Running task:", task)
    subprocess.run(command, shell=True)
