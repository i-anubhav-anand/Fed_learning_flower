# Fed_learning_flower


<h1>Introduction</h1>
How Flower can be used to build federated learning use cases based on existing machine learning projects.

<h2>Project Setup</h2>
Start by cloning the example project.
https://github.com/i-anubhav-anand/Fed_learning_flower

containing files:
'''bash 
 --client.py
 --client2.py
 -- server.py
 -- README.md
 --requiremtns.txt
 bash'''
 
 <h2>From Centralized To Federated<h2>
 This Project is based on the Deep Learning with PyTorch that uses the custom data of Coivd and Normal Patient to predict if the patient is having covid or not.
  
  You can simply start the centralized training as described
  Start the server in a terminal as follows:

  python server.py

  Now that the server is running and waiting for clients, we can start two clients that will participate in the federated learning process.
  To do so simply open two more   terminal windows and run the following commands.
  
  Start client 1 in the first terminal:

  python client.py

  Start client 2 in the second terminal:

  python3 client2.py
  
  You are now training a PyTorch-based CNN image classifier on Chest X-ray(Custom Data), federated across two clients.
