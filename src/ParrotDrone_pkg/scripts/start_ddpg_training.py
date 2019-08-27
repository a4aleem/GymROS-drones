import DDPG.filter_env
from DDPG.ddpg import *
import gc
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

gc.enable()



def main():


	rospy.init_node('parrotdrone_goto_qlearn',
                    anonymous=True, log_level=rospy.WARN)

	task_and_robot_environment_name = rospy.get_param(
        '/drone/task_and_robot_environment_name')

	env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    env = filter_env.makeFilteredEnv(env)
    rospy.loginfo("Gym environment done")
    agent = DDPG(env)

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('my_parrotdrone_openai_example')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    nepisodes = rospy.get_param("/drone/nepisodes")
    nsteps = rospy.get_param("/drone/nsteps")
    TEST = rospy.get_param("/drone/test")

    start_time = time.time()
    highest_reward = 0

    for episode in xrange(nepisodes):

    	rospy.logdebug("############### START EPISODE=>" + str(episode))
        cumulated_reward = 0
        done = False

        observation = env.reset()
        state = ''.join(map(str, observation))
        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot for nsteps

        # Train
        for i in xrange(nsteps):

        	rospy.logwarn("############### Start Step=>" + str(i))
            # Pick an action based on the current state
            action = agent.noise_action(state)

            rospy.logwarn("Next action is:%d", action)
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)
            agent.perceive(state,action,reward,observattion,done)
            rospy.logwarn(str(observation) + " " + str(reward))
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            # Make the algorithm learn based on the results
            rospy.logwarn("# state we were=>" + str(state))
            rospy.logwarn("# action that we took=>" + str(action))
            rospy.logwarn("# reward that action gave=>" + str(reward))
            rospy.logwarn("# episode cumulated_reward=>" +
                          str(cumulated_reward))
            rospy.logwarn(
                "# State in which we will start next step=>" + str(nextState))
            qlearn.learn(state, action, reward, nextState)

            if not (done):
                rospy.logwarn("NOT DONE")
                state = nextState
            else:
                rospy.logwarn("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
            rospy.logwarn("############### END Step=>" + str(i))
            #raw_input("Next Step...PRESS KEY")
            # rospy.sleep(2.0)


        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)


        rospy.logerr(("EP: " + str(episode + 1) +  "- Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))





        # Testing:
        if episode % 100 == 0 and episode > 100:
			total_reward = 0
			for i in xrange(TEST):
				observation = env.reset()
        		state = ''.join(map(str, observation))
				for j in xrange(nsteps):
					#env.render()
					action = agent.action(state) # direct action for test
					observation, reward, done, info = env.step(action)
					total_reward += reward
					if done:
						break
			ave_reward = total_reward/TEST
			rospy.logwarn("episode: " + str(episode) + "'Evaluation Average Reward:" + str(ave_reward))
			print 'episode: ',episode,'Evaluation Average Reward:',ave_reward

	rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(highest_reward) + "| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(
        reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))
    
    env.close()

if __name__ == '__main__':
    main()
