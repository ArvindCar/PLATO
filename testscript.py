from frankapy import FrankaArm

if __name__ == "__main__":
    fa = FrankaArm()
    fa.open_gripper()
    fa.goto_gripper(width=0.01, grasp=True)