PROMPT = """
You have been provided code for a memory agent. I want to extend its functionality. Right now, it stores memories user id wise. I want it to store user id and category wise. 
While storing a memory it should determine if a memory is personal, professional, or other. It should then save based on that. While retrieving memories it should determine what category the message(s) are from. 
Based on that, it should only retrieve relevant memories. Additionaly, there should be an interrupt before saving a memory. If the user inputs accept, the memory should get saved.
If the user inputs anything else, the memory should be rejected. Fix any other bugs/issues as well.
"""