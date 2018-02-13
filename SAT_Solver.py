"""
Carmi Rothberg
SAT Solver

"""

import numpy, scipy
from collections import defaultdict
from itertools import chain

#==============================================================================
#                               PART I
#==============================================================================


#==============================================================================
# Compute the co-occurrence matrix. The number of occurrences of word w and
# context word c = the number of (c,w) and (w,c) bigram in the corpus.
#==============================================================================

print("\n***\t\tPART I\t\t***\n\n")

data = open("dist_sim_data.txt","r").readlines()
data = [sent.split() for sent in data]

vocab = set()

context = defaultdict(lambda : defaultdict(int))

for sent in data:
    for i in range(len(sent)):
        word = sent[i]
        vocab.add(word)
        for j in chain(range(0, i), range(i+1, len(sent))):
            context[sent[i]][sent[j]] += 1

vocab = list(vocab)

# Matrix C
C = numpy.zeros((len(vocab), len(vocab)))
for i in range(len(vocab)):
    for j in range(len(vocab)):
        C[i, j] = (context[vocab[i]][vocab[j]])

#==============================================================================
# Multiply your entire matrix by 10 (to pretend that we see these sentences 10
# times) and then smooth the counts by adding 1 to all cells.
#==============================================================================
C *= 10
C += 1

#==============================================================================
# Compute the positive pointwise mutual information for each word w and
# context word c
#==============================================================================
PPMI = {}
word_prob = C.sum(axis=1)/C.sum()
cont_prob = C.sum(axis=0)/C.sum()
for w in range(len(vocab)):
    for c in range(len(vocab)):
        P_w_and_c = C[w, c]/C.sum()
        P_w = word_prob[w]
        P_c = cont_prob[c]
        PPMI[(w, c)] = max(numpy.log2(P_w_and_c/(P_w * P_c)), 0)
        
#==============================================================================
# Reweight your count matrix with the PPMI by multiplying them element-wise.
# Then compare the word vector for “dog” before and after PPMI reweighting.
#==============================================================================
print("Before PPMI reweighting, the word vector for 'dogs' is:")
for i in range(len(vocab)):
    print(vocab[i], end=': ')
    print(C[vocab.index("dogs")][i])

for i in range(len(vocab)):
    for j in range(len(vocab)):
        C[i,j] *= PPMI[(i,j)]

print("\nAfter PPMI reweighting, the word vector for 'dogs' is:")
for i in range(len(vocab)):
    print(vocab[i], end=': ')
    print(C[vocab.index("dogs")][i])

#==============================================================================
# At this point, we have a functional distributional semantic model. Let’s try
# to find similar word pairs. Compute the Euclidean distance between the
# following pairs:
#==============================================================================

# – women and men (human noun vs. human noun)
# – women and dogs (human noun vs. animal noun)
# – men and dogs (human noun vs. animal noun)
# – feed and like (human verb vs. human verb)
# – feed and bite (human verb vs. animal verb)
# – like and bite (human verb vs. animal verb)

word_pair_list = [('women','men'), ('women','dogs'), ('men','dogs'),
                  ('feed','like'), ('feed','bite'), ('like','bite')]

def euclidean_distance(matrix, w1, w2):
    vecSim = scipy.linalg.norm(matrix[vocab.index(w1)] - matrix[vocab.index(w2)])
    print('The Euclidean distance between "' + w1 + '" and "' + w2 + '" is ' +
          str(vecSim) + '.')

print("\nUsing the semantic model we've just created, note the Euclidean")
print("distances for the following word pairs:")
for (w1, w2) in word_pair_list:
    euclidean_distance(C, w1, w2)
#==============================================================================

#==============================================================================
# Decompose the matrix using SVD by using the command scipy.linalg.svd, using
# the commands given below. Then verify that you can recover the original
# matrix by multiplying U, E, and the conjugate transpose of V together.
#==============================================================================
U, E, V = scipy.linalg.svd(C, full_matrices = False)
E = numpy.matrix(numpy.diag(E)) # compute E
U = numpy.matrix(U) # compute U
V = numpy.matrix(V) # compute V
Vt = numpy.transpose(V) # compute conjugate transpose of V

#==============================================================================
# Reduce the dimensions to 3 to get word vectors.
#==============================================================================

reduced_C = C * V [:, 0 : 3]

#==============================================================================
# Compute the Euclidean distances of human/animal nouns/verbs again but on the
# reduced PPMI-weighted count matrix.
#==============================================================================

word_pair_list = [('women','men'), ('women','dogs'), ('men','dogs'),
                  ('feed','like'), ('feed','bite'), ('like','bite')]

print("\nFor the reduced matrix, note the new Euclidean distances between")
print("word pairs:")
for (w1, w2) in word_pair_list:
    euclidean_distance(reduced_C, w1, w2)


#==============================================================================
#                               PART II
#==============================================================================

print("\n\n***\t\tPART II\t\t***\n\n")

#==============================================================================
# Synonym Detection
#==============================================================================
print("Reading synonym data...")
syns = open("EN_syn_verb.txt","r").read().split('\n')[1:-1]
syns = [word_syn.split('\t') for word_syn in syns if word_syn.split('\t')[1] != '0']

syn_dict = defaultdict(list)
keys = []
vals = []

print("Creating synonym problems...")
for pair in syns:
    keys.append(pair[0])
    vals.append(pair[1])
    syn_dict[pair[0]].append(pair[1])

problems = []

for i in range(1000):
    problem = []
    word = numpy.random.choice(keys)
    syn = numpy.random.choice(syn_dict[word])
    problem.append(word)
    problem.append(syn)
    while (len(problem) < 6):
        other = numpy.random.choice(vals)
        if other not in syn_dict[word]:
            problem.append(other)
    problems.append(problem)

print("Reading Google vectors...")
f = open("GoogleNews-vectors-rcv_vocab.txt").readlines()
w2v = {}
for line in f:
    w2v[line.split()[0]] = line

print("Reading COMPOSES vectors...")
f = open("EN-wform.w.2.ppmi.svd.500.rcv_vocab.txt").readlines()
composes = {}
for line in f:
    composes[line.split()[0]] = line

print("\nDone!\n")

def euc_dist(vec_data, w1, w2):
    if w1 in vec_data and w2 in vec_data:
        line1 = numpy.matrix([float(val) for val in vec_data[w1].split()[1:]])
        line2 = numpy.matrix([float(val) for val in vec_data[w2].split()[1:]])
        
        vecSim = scipy.linalg.norm(line1 - line2)
        
        return vecSim
    else:
        return 10

def cos_dist(vec_data, w1, w2):
    if w1 in vec_data and w2 in vec_data:
        line1 = numpy.array([float(val) for val in vec_data[w1].split()[1:]])
        line2 = numpy.array([float(val) for val in vec_data[w2].split()[1:]])
        
        prod_w1_w2 = numpy.dot(line1, line2)
        sqrt_w1_sq = numpy.sqrt(sum([val*val for val in line1]))
        sqrt_w2_sq = numpy.sqrt(sum([val*val for val in line2]))
        
        vecSim = (prod_w1_w2)/(sqrt_w1_sq * sqrt_w2_sq)
        
        return 1-vecSim
    else:
        return 1

def solve_syn_problems(problems, vec_data, method):
    solved = 0
    answers = []
    for problem in problems:
        solved+=1
        if solved%200 == 0:
            print(str(solved) + " solved!")
        word = problem[0][3:]
        options = [option[3:] for option in problem[1:]]
        answer = min(options, key=lambda option: method(
                vec_data, word, option))
        answers.append(['to_'+word, 'to_'+answer])
    return answers

def eval_answers(answers):
    correct = 0
    incorrect = 0
    for [word, answer] in answers:
        if answer in syn_dict[word]:
            correct += 1
        else:
            incorrect += 1
    print(str(correct) + " correct out of " + str(correct+incorrect))

print("Let's now attempt to solve the synonym problems created above.")
print("\nUsing Euclidean distance and Google data:")
w2v_euc_answers = solve_syn_problems(problems, w2v, euc_dist)
eval_answers(w2v_euc_answers)

print("\nUsing Euclidean distance and COMPOSES data:")
composes_euc_answers = solve_syn_problems(problems, w2v, euc_dist)
eval_answers(composes_euc_answers)

print("\nUsing cosine distance and Google data:")
w2v_cosine_answers = solve_syn_problems(problems, w2v, cos_dist)
eval_answers(w2v_cosine_answers)

print("\nUsing cosine distance and COMPOSES data:")
composes_cosine_answers = solve_syn_problems(problems, composes, cos_dist)
eval_answers(composes_cosine_answers)

#==============================================================================
# SAT Questions
#==============================================================================

print("\nReading SAT data...")
sat_data = open("SAT-package-V3.txt","r").readlines()[41:]

print("Creating SAT problems...")
sat_questions = []
for i in range(0, len(sat_data), 9):
    q = sat_data[i+2].split()[:2]
    choices = []
    for k in range(3, 8):
        choices.append(sat_data[i+k].split()[:2])
    correct = sat_data[i+8][0]
    sat_questions.append([q, choices, correct])

print("Done!\n")

def sat_solve(questions, vec_data, method):
    answers = []
    correct = 0
    incorrect = 0
    for q in questions:
        wA = q[0][0]
        wB = q[0][1]
        options = []
        for [w1, w2] in q[1]: #for option in options
            dist = method(vec_data, w2, w1) - method(vec_data, w2, wA) + method(vec_data, w2, wB)
#            dist = (method(vec_data, w2, w1) * method(vec_data, w2, wB)) / method(vec_data, w2, wA)
            options.append(dist)
        actual_answer = q[2]
        key = options.index(min(options))
        my_answer = chr(97 + key)
        if my_answer == actual_answer:
            correct += 1
        else:
            incorrect += 1
        answers.append((wA + ' : ' + wB + ' :: ' + str(q[1][key][0]) + ' : ' + str(q[1][key][1])))
    print(str(correct) + ' correct out of ' + str(correct+incorrect))
    return answers

print("Let's now attempt to solve the SAT problems created above.")
print("\nUsing Euclidean distance and Google data:")
w2v_euc_answers = sat_solve(sat_questions, w2v, euc_dist)

print("\nUsing Euclidean distance and COMPOSES data:")
composes_euc_answers = sat_solve(sat_questions, w2v, euc_dist)

print("\nUsing cosine distance and Google data:")
w2v_cosine_answers = sat_solve(sat_questions, w2v, cos_dist)

print("\nUsing cosine distance and COMPOSES data:")
composes_cosine_answers = sat_solve(sat_questions, composes, cos_dist)