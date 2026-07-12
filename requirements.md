Technical Exercise in Machine Learning and Robotics v2
Introduction
The goal of this exercise is to introduce the scope and flavour of work we need to do
at Imitation Machines as we move forward to our goal of building an integrated
software platform that makes imitation and reinforcement learning technologies
accessible for practical robot learning applications and deployment.
We hope the exercise will be a curious learning experience that connects the
established robotics toolboxes with the emerging open source ecosystem of robot
learning, testing coding and system design skills along the way.
The tasks in this exercise are practical, instead of looking for a “right answer” we aim
to explore your approach, problem-solving, goal-oriented development, and the
structure of the solution. You can use any existing open source libraries, rely on
ChatGPT, copilot, etc., and in general use any tool available, as you would in the
normal course of work.
Step 1: Simulator Setup
Set up a robotics simulator and import a robot model into it. Depending on your
preference and available hardware you can use MuJoCo (lightweight,
https://github.com/google-deepmind/mujoco
_
menagerie) or NVIDIA Isaac Sim
(https://github.com/isaac-sim/IsaacSim, requires an NVIDIA GPU and Ubuntu 22.04)
Next, pick a robot and import it into the simulator. Pick any robot you like. One option
that is likely to be compatible with the Step 3 of this exercise is the SO-100/101 robot
arm, but choose whichever robot you’ll have the most fun working with!
Step 2: ROS2 Integration
Connect the robot in the simulator via ROS2 Humble to a Python script so that you
can control the robot’s movements from the code, and read the robot's sensors and
telemetry. Write a script to make the robot execute a simple motion, such as tracing
an imaginary circle in the air, walking a circle (if it’s a mobile robot), raising the arm
and waiving, or anything else of that nature.
Confidentiality Notice: This document and the information contained in it are confidential and intended solely for the individual
or entity to whom it is addressed. Unauthorized use, disclosure, or distribution of any portion of this document is prohibited.
Step 3: Conversion into LeRobot Format
Next, let’s get familiar with the LeRobot repository and the way they structure their
datasets (https://github.com/huggingface/lerobot). The goal is to record a minimal
dataset (camera, minimal telemetry, ~10 episodes) in their format and visualise it
using LeRobot’s internal dataset visualisation tool (to confirm the format is correct).
First, dive into their Dataset class definition to get the general gist of the structure
(https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/lerobot
dat
_
aset.py#L598). Download some existing datasets of theirs to explore the folder
structure and run your local instance of their dataset visualisation tool explore
(https://github.com/huggingface/lerobot/tree/main?tab=readme-ov-file#visualize-da
tasets) some pre-recorded episodes.
Update your script from Step 2 to record robot telemetry and the camera feed(s)
from the simulator, and store it in LeRobot format. Feel free to just reuse existing
dataset and modify the files instead of writing the whole dataset conversion code (a
quick and hacky solution is ok for this part, as long as it works). Run their
visualisation tool on your new dataset and present screenshots and results.
Step 4: Train an ACT model with LeRobot
Once you’ve collected several episodes like this, it should now be possible to just use
existing LeRobot scripts to train an imitation learning policy that will l earn to execute
the demonstrated behaviour! For example if you have created a routine that made
the robot wave “hello”
, with this you can turn these demonstrations into an
autonomous behavior policy.
Of course learning to replicate a robot motion that you have already programmed has
no practical benefit, but imagine if instead of hardcoded demonstration you would
have a human demonstrating any behaviour they want… That’s the potential of
imitation learning for robot automation, and the power that Imitation Machines aims
to bring to robot end-users around the world!
Train a policy for an hour, it does not actually learn the behaviour successfully, for
this part of the exercise it is sufficient that the model starts to learn on your dataset.
While the model is training, explain in your readme/report which learning metrics you
are tracking, why they are important and what you can tell from observing them.
Deploy your newly trained policy on the robot and record a video of the simulator
screen showing your policy running. This concludes the exercise!
Confidentiality Notice: This document and the information contained in it are confidential and intended solely for the individual
or entity to whom it is addressed. Unauthorized use, disclosure, or distribution of any portion of this document is prohibited.
Details
You will have one week to work on this exercise starting at the date & time of your
choice. The use of artificial intelligence tools is allowed and encouraged. Please do
not use any human help though, the submitted work should be completed by you.
If you will have any questions along the way please feel free to send an email to
ilya@imitationmachines.com and I will endeavour to reply promptly.
It is highly recommended to use Ubuntu 22.04/24.04 in your work (for Isaac Sim
22.04 is a requirement, but also other frameworks and packages have a higher
chance of working out of the box on Ubuntu). Getting everything to run on other
Linuxes, on a Mac or a Windows most likely will prove to be frustratingly difficult.
We tried to keep a balance between the exercise being a useful learning experience,
having practical relevance, and having reasonable time requirements, however it
might be that some of the steps are more time consuming than we anticipated.
Please do not be discouraged from submitting partial solutions, as mentioned before
– our primary goal here is to explore your approach and problem-solving and give a
taste for the kind of work we do at Imitation Machines.
Delivery
Please submit the codebase of your solution (if sharing via github or alike please use
a private repository, so that we wouldn’t have to re-design this exercise too often).
Provide instructions on how to run your code, note and explain key design decisions
you made and the reasons for them, accompanied by figures and illustrations where
necessary. A repository readme.md is sufficient, you do not need to create a
separate PDF report, but you can, if you prefer to.
Along with your submission please provide a video recording of your screen
demonstrating the steps and the final solution step by step, and any other elements
you would like to demonstrate.
Have fun, and good luck!