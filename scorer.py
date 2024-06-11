def scorer_mental(groud_truth,responses):
    import numpy as np
    scores=[]
    for gd, ans in zip(groud_truth, responses):
        if ans.strip() != "-1":
            scores.append(abs(float(gd)-float(ans)))
    mae = np.mean(scores)
    return mae

                    