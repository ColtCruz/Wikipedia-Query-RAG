#DocumentOverview
For this project, I picked the Wikipedia article on the Popol Vuh. It’s one of the most important texts in Mayan mythology, written by the Kʼicheʼ people of Guatemala. It covers the creation of the world, the adventures of the Hero Twins, and even royal lineages. It's a dense, symbolic narrative that holds a lot of cultural weight, which made it a solid pick to test how well a RAG system could handle meaningful, structured content.

#DeepDiveQuestions
Q1: Why does chunk size matter in a RAG pipeline?
Chunk size controls how much text the model gets at once. Too small, and it misses context. Too big, and it might get overwhelmed or slow down. The goal is to give it just enough to work with—clear, digestible context that leads to better answers.

Q2: How does chunk overlap affect retrieval accuracy?
Overlap keeps the flow of ideas intact. When sections of text spill over between chunks, overlap makes sure nothing important falls through the cracks. It helps the model stay on track.

Q3: What is embedding dimensionality and why is it important?
It’s the size of the vector that represents each chunk. Bigger dimensions usually mean better detail and understanding, but they also eat up more memory and processing time. It’s a tradeoff between richness and performance.

Q4: How does FAISS determine similarity between chunks?
FAISS uses distance metrics (like cosine similarity) to compare the question's embedding to each chunk. The closer they are in vector space, the more relevant the chunk is likely to be.

Q5: Why is prompt design important for question answering?
Prompts steer the model. A clear, specific prompt sets up a better response. A vague one? You’ll probably get vague answers. It’s all about asking the right way.

#ChunkingExperiments
I tried a few different settings to see what gave the best results. Starting with a small chunk size of 100 and minimal overlap, the answers felt too short and shallow. Then I bumped it to 300 with a 50-token overlap—huge improvement. The context was just right, and the responses were way more complete and on point. I tested bigger sizes like 500 and 1000, but they started dragging in too much info and sometimes got sidetracked. Overall, chunk_size = 300 with chunk_overlap = 50 was the sweet spot.

#ResultsAndTakeaways
Once I realized I hadn’t saved the updated document (rookie mistake), everything clicked. After re-running with the proper text, the answers finally matched the material. For example, it correctly identified the Hero Twins as Hunahpu and Xbalanque, instead of giving weird, unrelated names like before.

#SuggestionsForImprovement
Clean the input text: Remove things like citation numbers and references before chunking.

Multi-source support: Let the RAG system pull from multiple docs or URLs for a broader base of knowledge.

Use section headers: Add Wikipedia section titles to each chunk to organize context better.

Add confidence checks: Maybe include a feedback system or show confidence levels with each answer.

