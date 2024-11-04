# Machine Learning Interview Guide
_Last updated: 2024-11-05_  
_Owner: Yu Sun_


This chapter is focusing on ML coding questions. We will list some of the most popular ML coding questions and try to solve them step by step.

- **Task Status Legend**  
    - ☐ Not Started  
    - ➤ In Progress  
    - ☑ Completed  
  
- **Parameters**  
    - n: # of instances, or # of rows in a matrix_1  
    - d: # of features, or # of columns in a matrix_1  
    - m: # of rows in a matrix_2
    - k: # of columns in a matrix_2
  
- **Template**  
    - Please use the template in [0000_template.ipynb](ipynb_codes/0000_template.ipynb) to implement your code.
    - Upate your the code, customized unittest with at least these 3 types of test cases:
        1. Dimension mismatch and corner cases
        2. Normal cases with all possible input types
        3. Extreme cases
  
## Linear Algebra/Probability/Statistics

<table>
  <thead>
    <tr>
      <th rowspan="3">Status</th>
      <th rowspan="3">Question Name</th>
      <th rowspan="3">Source</th>
      <th rowspan="3">Ipynb Implementation</th>
      <th colspan="4">Complexity</th>
    </tr>
    <tr>
      <th colspan="2">Training</th>
      <th colspan="2">Inference</th>
    </tr>
    <tr>
      <th>Time</th>
      <th>Space</th>
      <th>Time</th>
      <th>Space</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>☐</td>
      <td>0001 Matrix Times Vector</td>
      <td><a href="https://www.deep-ml.com/problem/Matrix%20times%20Vector">deep-ml-0001-Matrix times Vector</a></td>
      <td><a href="ipynb_codes/0001_matrix_times_vector.ipynb">0001_matrix_times_vector.ipynb</a></td>
      <td>O(nd)</td>
      <td>O(d)</td>
      <td>O(nd)</td>
      <td>O(d)</td>
    </tr>
  </tbody>
</table>







## Traditional ML

## Deep Learning

## CV

## NLP and LLM

## RL




## References

### Top public ML problems websites

- [deep-ml.com](https://deep-ml.com)

### Top Starred Repositories for Machine Learning Interviews

Here we chose public repos with keywords "machine learning interview" in GitHub with over 1k stars.

| Repository                                                                                      | Stars | Last Update | Summary                                                                                                                                                                                                                                         |
|-------------------------------------------------------------------------------------------------|-------|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [khangich/machine-learning-interview](https://github.com/khangich/machine-learning-interview)   | 9.7k  | 2023-08-31  | A comprehensive ML interview repository covering LeetCode-style questions, ML Q&A, and real-world ML system design examples.                                                                                                                    |
| [aaronwangy/Data-Science-Cheatsheet](https://github.com/aaronwangy/Data-Science-Cheatsheet)     | 5k    | 2023-03-15  | A simple, unfinished data science cheatsheet covering limited foundational ML concepts.                                                                                                                                                        |
| [alirezadir/Machine-Learning-Interviews](https://github.com/alirezadir/Machine-Learning-Interviews) | 4.7k  | 2024-03-05  | A complete ML interview guide with sections on General Coding, ML Coding, ML System Design, ML Theory, and Behavioral aspects, forming the basis of this repository's structure.                                                              |
| [andrewekhalel/MLQuestions](https://github.com/andrewekhalel/MLQuestions)                       | 3k    | 2024-05-22  | Contains 65 foundational ML questions and 13 NLP-related questions, with external references for answers rather than direct solutions.                                                                                                        |
| [rbhatia46/Data-Science-Interview-Resources](https://github.com/rbhatia46/Data-Science-Interview-Resources) | 2.8k  | 2024-08-07  | Contains numerous foundational questions on Data Science and ML, with answers linked to external blogs or videos.                                                                                                                             |
| [DarLiner/Algorithm_Interview_Notes-Chinese](https://github.com/DarLiner/Algorithm_Interview_Notes-Chinese) | 2.2k  | 2018-11-03  | Comprehensive Chinese-language interview questions covering programming, data structures, algorithms, math, CV, NLP, ML, DL, and includes practical coding exercises.                                                                          |
| [Sroy20/machine-learning-interview-questions](https://github.com/Sroy20/machine-learning-interview-questions) | 1.5k  | 2019-05-19  | A set of over a hundred ML, DL, and math interview questions, without accompanying answers.                                                                                                                                                    |
| [zhengjingwei/machine-learning-interview](https://github.com/zhengjingwei/machine-learning-interview) | 1.3k  | 2019-09-26  | A Chinese-language repository with over a hundred ML interview questions, mostly without solutions.                                                                                                                                            |
| [devAmoghS/Machine-Learning-with-Python](https://github.com/devAmoghS/Machine-Learning-with-Python) | 1.2k  | 2024-08-04  | Extensive ML coding examples, complete with data and Python code for hands-on practice.                                                                                                                                                       |
| [geektutu/interview-questions](https://github.com/geektutu/interview-questions)                 | 1.1k  | 2024-08-06  | Contains some Chinese-language ML interview questions and a few practical clustering algorithm examples.                                                                                                                                      |
