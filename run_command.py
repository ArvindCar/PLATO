def run_command(act, feature, deltas, fa):
    return
    if act == 'Go-to':
        pose = fa.get_pose()
        pose.translation = feature + deltas
        fa.goto_pose(pose)
        
    if act == 'Grasp':
        if feature == 0:
            fa.open_gripper()
        elif feature == 1:
            fa.close_gripper()
        
    return
