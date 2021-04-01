# Message Distortion in Information Cascades

This repository contains the data and the experiments and the 
instructions of the paper "Message Distortion in Information Cascades" ([which you can read clicking here](https://arxiv.org/pdf/1902.09197.pdf)):
  
  ~~~bibtex
 @inproceedings{horta_ribeiro_message_2019,
 author={Ribeiro, Manoel Horta and Gligor\'ic, Kristina and West, Robert},
 title={Message Distortion in Information Cascades},
 booktitle={Proceedings of the 2019 World Wide Web Conference},
 year={2019},
 }
 ~~~

[Chek out the accompanying website which allows you to visualize the data.](https://epfl-dlab.github.io/mdic/)
# Data

You may find the data in the data folder `./data` (duh). 

We make the data available in two formats: a `.csv` and `.graphml`. 
The latter is the format used in the analysis of the data (for convenience).


| Field         | Null in Root|  Description |
|---------------|-------------|--------------|
| node_id       | No | Unique identifier of the node. |
| level         | No | Summarization level: (0: original abstract, 1: ~1024 chars, 2: ~512 chars, 3: ~256 chars, 4: ~124 chars, 5: ~64 chars) |
| branch        | No | Source note used in this information cascade. |
| question      | No  | `node_id` of the text used as reference for the summarization, in case of root nodes, it is the same as `node_id` |
| Topic         | No | Topic of the paper summarized (breast, cardio, immunization, diet) |
| Answer        | No | Original abstract in the case of the root, summary otherwise. |
| Age           | Yes | Age range of the worker which summarized the paper (18-24/24-39/40-60/60+)
| Education     | Yes | Education level of the worker which summarized the paper (Some High School, High School, Some College, College) |
| Gender        | Yes | Gender of the worker which summarized the paper (male, female) |
| Qualification | Yes | Performance on qualification test (float, 0-1) |
| WorkerId      | Yes | Unique worker identifier as provided by amazon mt |
| Doggos_crowd  | No | Dictionary containing the values for facts in each category. `{"Coarse":{"Coarse_category1":["Val1", "Val2", ...] ...}, "Fine":{"Fine_category1":["Val1"} ...}`
| Doggos_text   | No | Dictionary containing the text for facts of each sub category. Null in non-root. `{"Fine_category1": "Text1", "Fine_category2": "Text2", ... `}|
| Tagging       | No | In the `csv` files, this is a Dictionary, similar to doggos crowd, containing the keyphrases associated with each subcategory. `{"Coarse": {"keyword1": {"Course_category1", ...}, ... }, "Fine": {"keyword1": {"Fine_category1", ...}, ... }`. For the `graphml` files, this is actually a python object with this dictionary, and a bunch of helper functions to calculate the difference in keywords across hops.|


# Code

All the analysis performed may be found in the `analyses.ipynb` notebook.

To install all requirements simply run 

    pip install -r requirements.txt 

