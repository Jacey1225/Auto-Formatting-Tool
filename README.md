AI PROJECT MANAGEMENT ASSISTANT - SLACK BOT COMMUNICATOR HANDLES COMMONLY KNOWN PROJECT-BASED QUERIES BASED ON CURRENTLY AVAILABLE PROJECT DATA

- responsive web design used to organize and keep track of project insights such as performance reports, budget analysis, and project progress/timeline
- AI slack bot used to communicate with project members on tasks, or miscellaneous queries such as budget, progress, status, performance, efficiency, etc. info requests. 

This bot will use term-frequency(TF-IDF) analyzers and semantic vectors(FastText) to predict the best translation that calls a function to handle said requests. In order to reduce computing cost, most requests will be handled via out-soured data through web-scraping and parsing. 


Pre-Fetched Data Approach and Training:
    DataSet Processing:
        - Queries(Features): Tokenize, remove stopwords, lemmatize, POS tag, chunk with SpaCy, capture entities with NER
        - Raw Sources(features): Tokenize(maintain original structure)
        - Responses(Labels): Tokenize, lemmatize, chunk with SpaCy, capture entities with NER

    Vectorization:
        - Train BERT on all features and labels, average out each word to reduce complexity

    BERT Training: 
        - BERT will be trained to understand common patterns between the queries, the raw data, and how they interact with the labels. It
        would capture the syntactical meaning of responses, the structure of speech with POS recognition, and refer to the raw sources as a retrieval corpus

    




# Auto Formatting Application
