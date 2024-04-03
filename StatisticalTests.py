import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance, mannwhitneyu, fisher_exact

def power_law(k_min, k_max, y, gamma):
    return ((k_max**(-gamma+1) - k_min**(-gamma+1))*y  + k_min**(-gamma+1.0))**(1.0/(-gamma+1.0))

def generate(N, k_min, k_max, gamma):
    center = np.zeros(N)
    y = np.zeros(N)
    for i in range(N):
        y[i] = np.random.uniform(0, 1)
        center[i] = int(power_law(k_min, k_max, y[i], gamma))
        center[i] -= 1
    center = sorted(center, reverse = True)
    return center

ds0 = [9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
ds1 = [168, 132, 84, 40, 34, 34, 32, 24, 18, 14, 12, 12, 12, 8, 0, 0]
ds2 = [87, 26, 24, 24, 12, 12, 11, 8, 7, 6, 6, 6, 4, 2, 2, 0]
ds3 = [38, 9, 8, 8, 8, 7, 7, 5, 4, 4, 3, 3, 2, 1, 1, 0]
ds4 = [36, 16, 12, 8, 7, 5, 4, 3, 3, 2, 1, 1, 1, 0, 0, 0]

def KolmogorovSmirnov(ds, T):
    cnt0 = 0
    cnt1 = 0
    for i in range(T):
        dsr = generate(len(ds), 1, ds[0] + 1, 1.1)
        ks_statistic, p_value = ks_2samp(dsr, ds)
        if p_value < 0.05:
            cnt0 += 1
        else:
            cnt1 += 1
    return (cnt0, cnt1)

def MannWhitneyU(ds, T):
    cnt0 = 0
    cnt1 = 0
    for i in range(T):
        dsr = generate(len(ds), 1, ds[0] + 1, 1.1)
        u_statistic, p_value = mannwhitneyu(dsr, ds, alternative='two-sided')
        if p_value < 0.05:
            cnt0 += 1
        else:
            cnt1 += 1
    return (cnt0, cnt1)

def FisherExact(ds, T):
    cnt0 = 0
    cnt1 = 0
    for i in range(T):
        dsr = generate(len(ds), 1, ds[0] + 1, 1.1)
        data1 = np.array(dsr)
        data2 = np.array(ds)

        threshold = np.median(np.concatenate((data1, data2)))

        data1_category = data1 > threshold
        data2_category = data2 > threshold

        a = np.sum(data1_category)
        b = np.sum(data2_category)
        c = len(data1) - a
        d = len(data2) - b

        table = [[a, b], [c, d]]

        odds_ratio, p_value = fisher_exact(table)

        if p_value < 0.05:
            cnt0 += 1
        else:
            cnt1 += 1
    return (cnt0, cnt1)

def Permutation(ds, T):
    cnt0 = 0
    cnt1 = 0
    for i in range(T):
        dsr = generate(len(ds), 1, ds[0] + 1, 1.1)
        combined = np.concatenate([dsr, ds])

        n_permutations = 10000

        observed_diff = np.mean(ds) - np.mean(dsr)

        count = 0

        for _ in range(n_permutations):
            np.random.shuffle(combined)


            new_data1 = combined[:len(ds)]
            new_data2 = combined[len(ds):]
            new_diff = np.mean(new_data1) - np.mean(new_data2)

            if np.abs(new_diff) >= np.abs(observed_diff):
                count += 1

        p_value = count / n_permutations

        if p_value < 0.05:
            cnt0 += 1
        else:
            cnt1 += 1
    return (cnt0, cnt1)

print(KolmogorovSmirnov(ds0, 100))
print(KolmogorovSmirnov(ds1, 100))
print(KolmogorovSmirnov(ds2, 100))
print(KolmogorovSmirnov(ds3, 100))
print(KolmogorovSmirnov(ds4, 100))

print(MannWhitneyU(ds0, 100))
print(MannWhitneyU(ds1, 100))
print(MannWhitneyU(ds2, 100))
print(MannWhitneyU(ds3, 100))
print(MannWhitneyU(ds4, 100))

print(FisherExact(ds0, 100))
print(FisherExact(ds1, 100))
print(FisherExact(ds2, 100))
print(FisherExact(ds3, 100))
print(FisherExact(ds4, 100))

print(Permutation(ds0, 100))
print(Permutation(ds1, 100))
print(Permutation(ds2, 100))
print(Permutation(ds3, 100))
print(Permutation(ds4, 100))