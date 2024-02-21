# Identifying the Unfolding of New Events in a Narrative (presented at The 5th Workshop on Narrative Understanding 2023) 
We introduce the new task of identifying new events as they unfold in the narrative. In our definition of the event, the verb is the central element that represents a relation/happening that engages its dependencies such as subject, object, or oblique nominals. Meanwhile, we define an event as new if it provides novel information to the reader with respect to the discourse (discourse-new) and if such information can not be inferred through commonsense. We annotated a complete dataset of personal narratives, SEND (Ong et al. 2019), with new events at the sentence level using human annotators. The dataset consists of 193 narratives from 49 subjects, collected by asking each narrator to recount 3 most positive and 3 most negative experiences of her/his life. We believe this task can be a novel and challenging task in narrative understanding and can facilitate and support other tasks in natural language understanding, human-machine dialogue, and natural language generation.

## How to cite us 
```
@inproceedings{mousavi-etal-2023-whats,
    title = "What{'}s New? Identifying the Unfolding of New Events in a Narrative",
    author = "Mousavi, Seyed Mahed  and
      Tanaka, Shohei  and
      Roccabruna, Gabriel  and
      Yoshino, Koichiro  and
      Nakamura, Satoshi  and
      Riccardi, Giuseppe",
    editor = "Akoury, Nader  and
      Clark, Elizabeth  and
      Iyyer, Mohit  and
      Chaturvedi, Snigdha  and
      Brahman, Faeze  and
      Chandu, Khyathi",
    booktitle = "Proceedings of the The 5th Workshop on Narrative Understanding",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.wnu-1.1",
    doi = "10.18653/v1/2023.wnu-1.1",
    pages = "1--10",
    abstract = "Narratives include a rich source of events unfolding over time and context. Automatic understanding of these events provides a summarised comprehension of the narrative for further computation (such as reasoning). In this paper, we study the Information Status (IS) of the events and propose a novel challenging task: the automatic identification of new events in a narrative. We define an event as a triplet of subject, predicate, and object. The event is categorized as new with respect to the discourse context and whether it can be inferred through commonsense reasoning. We annotated a publicly available corpus of narratives with the new events at sentence level using human annotators. We present the annotation protocol and study the quality of the annotation and the difficulty of the task. We publish the annotated dataset, annotation materials, and machine learning baseline models for the task of new event extraction for narrative understanding.",
}
