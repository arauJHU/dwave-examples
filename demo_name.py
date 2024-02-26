# Copyright 2024 Aaditya Rau
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dimod import Real, Binary, Integer, ConstrainedQuadraticModel 
from dwave.system import LeapHybridCQMSampler
import numpy as np

import matplotlib.pyplot as plt

def ElasticityModel(elastic_strain):

    '''
    Elastic modulus can be changed here!
    '''

    E = 5 # Young's Modulus (Pa)

    we = 0.5 * E * elastic_strain**2    # Elastic energy
    dwe = E * elastic_strain            # Stress
    ddwe = E                            # Elastic tangent - just modulus in this case

    return we, dwe, ddwe

def HardeningModel(Q):

    # Only consider isotropic hardening

    '''
    Material properties: sigmaY = initial yield stress, H = hardening modulus
    '''

    sigmaY = 2.5e-4
    H = 1.e-1

    wp = sigmaY * Q + 0.5 * H * Q**2
    dwp = sigmaY + H * Q
    ddwp = H
    
    return wp, dwp, ddwp

def CheckYielding(sigma, Q):

    _, sigmaY_current, _ = HardeningModel(Q)

    tol = 1.e-12 # tolerance for finite precision

    # This is the small strain yield condition in 1D
    if (abs(sigma) > sigmaY_current + tol):
        yield_flag = True
    else:
        yield_flag = False

    return yield_flag

def AssembleWork(epsilon, epsP):

    we, dwe, ddwe = ElasticityModel(epsilon - epsP)
    wp, dwp, ddwp = HardeningModel(abs(epsP))

    # Signum function that seems to not cause problems at the yield point
    if (epsP >= 0):
        sgnepsP = 1
    else:
        sgnepsP = -1

    Lag = we + wp 
    dLag = -1 * dwe + dwp * sgnepsP
    ddLag = ddwe + ddwp

    return Lag, dLag, ddLag

def IncrementalUpdateNewton(epsilon, q):

    epsP = q[0]

    tol = 1.e-12 # tolerance for newton iteration
    converged = False

    while not converged:

        _, dLag, ddLag = AssembleWork(epsilon, epsP)

        depsP = -dLag/ddLag

        epsP += depsP

        if (abs(depsP) < tol):
            converged = True
            
    return epsP

def IncrementalUpdateQuantum(epsilon, q, N_digits):

    # Get coefficients and center for Taylor series expansion of the dissipation function
    epsP_old = q[0]

    # Factor used to control the floating point representation
    if (abs(epsP_old) > 0):
        scaling = np.ceil(np.log10(abs(epsP_old)))
    else:
        scaling = -5

    Lag, dLag, ddLag = AssembleWork(epsilon, epsP_old)

    # Make some digits for input into the quantum computer
    digits = [Binary(f"d_{digit}") for digit in range(0, N_digits)]

    # Model that will be optimized
    cqm = ConstrainedQuadraticModel()
    cqm.set_objective(TaylorExpansion(digits, epsP_old, Lag, dLag, ddLag, scaling))

    # Instantiate solver
    sampler = LeapHybridCQMSampler()

    # Get a set of candidate solutions and filter for the feasible ones
    sampleset = sampler.sample_cqm(cqm)
    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)

    best = feasible_sampleset.first.sample

    return bin_to_dec(list(best.values()), scaling)

def TaylorExpansion(digits, epsP_old, Lag, dLag, ddLag, scaling):

    x = bin_to_dec(digits, scaling)

    # Taylor expansion about the old state
    TaylorPoly = Lag + dLag * (x - epsP_old) + 0.5 * ddLag * (x - epsP_old)**2

    return TaylorPoly

def bin_to_dec(digits, scaling):

    x = 0
    exponent = 0

    for digit in digits:
        x += digit * 2**exponent
        exponent -= 1

    x = x * 10**scaling

    return x

def plastic_update_1D(epsilon, qOld, quantumFlag):

    epsP_old = qOld[0] # Old plastic strain
    Q_old = qOld[1] # Old plastic eq. strain

    epsE_pre = epsilon - epsP_old # elastic predictor

    _, sigma_pre, _ = ElasticityModel(epsE_pre)

    yield_flag = CheckYielding(sigma_pre, Q_old)

    if yield_flag:
        
        # Let's do some plasticity!
        if (quantumFlag):
            
            # Control precision of quantum computing calculation here
            N_digits = 8

            # Do the update using quantum computing to do minimization
            epsP_new = IncrementalUpdateQuantum(epsilon, qOld, N_digits)

        else:

            # Traditional Newton solve of variational update
            epsP_new = IncrementalUpdateNewton(epsilon, qOld)

        _, sigma, _ = ElasticityModel(epsilon - epsP_new)

        q = np.zeros(np.size(qOld))
        q[0] = epsP_new
        q[1] = abs(epsP_new)

    else:
        # No plasticity so return the predictor and old internal var. state
        sigma = sigma_pre
        q = qOld

    return sigma, q


if __name__ == '__main__':

    # Quantum or classical solve?

    quantumFlag = True

    # Initialize containers
    N_samples = 66
    strain_final = 6.5e-5
    strain = np.linspace(0, strain_final, N_samples)
    q = np.zeros((2, N_samples + 1)) # + 1 because of initial condition
    stress = np.zeros((N_samples, 1))

    for iter in range(N_samples):

        stressNew, qNew = plastic_update_1D(strain[iter], q[:, iter], quantumFlag)
        stress[iter] = stressNew
        q[:, iter + 1] = qNew

    np.savetxt('stress_quantum.txt', stress, delimiter=',')
    # np.savetxt('strain_newton.txt', stress, delimiter=',')
    
    plt.scatter(100 * strain, stress)
    plt.xlabel('Strain (%)')
    plt.ylabel('Stress (Pa)')
    plt.savefig('stress-strain.png', bbox_inches='tight')

    # print(np.ceil(np.log10(abs(q[1, -1]))))


# def obj_func(digits, a):

#     '''
#     Assume digits are ordered such that first digit corresponds to 2^0
#     second digit corresponds to 2^-1, and so on
#     '''
#     scaling = -4
#     x = bin_to_dec(digits, scaling)
#     # exponent = 0
#     # x = 0
#     # for digit in digits:
#     #     x += digit * 2**exponent
#     #     exponent -= 1

#     obj_func_val = (x - a)**2 
#     return obj_func_val

# N_digits = 3
# digits = [Binary(f"d_{digit}") for digit in range(0, N_digits)]

# cqm = ConstrainedQuadraticModel()

# a = .5e-4
# cqm.set_objective(obj_func(digits, a))

# sampler = LeapHybridCQMSampler()

# sampleset = sampler.sample_cqm(cqm)
# feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)   
# print("{} feasible solutions of {}.".format(len(feasible_sampleset), len(sampleset)))

# best = feasible_sampleset.first.sample

# print(best)
# print(bin_to_dec(list(best.values()), -4))
