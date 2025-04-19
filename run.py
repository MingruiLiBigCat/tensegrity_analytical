import tr_env_gym
import os
from get_action import TensegrityStructure
from planning import COM_scheduler
def run():
    env = tr_env_gym.tr_env_gym(render_mode="None",
                                        xml_file=os.path.join(os.getcwd(),"3prism_jonathan_steady_side.xml"),
                                        is_test = False,
                                        desired_action = "straight",
                                        desired_direction = 1,
                                        terminate_when_unhealthy = True)
    scheduler = COM_scheduler(1, 0)

if __name__ == "__main__":
    run()

    
    #For step: env.step(action)
    #For get_obs: tensegrity.update_position_from_env(env)
    #For get_COM: scheduler.get_COM(x0, y0, x1, y1, x2, y2, x3, y3)
    