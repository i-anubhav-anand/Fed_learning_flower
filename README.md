# Federated_learning_flower


<h1>Introduction</h1>
How Flower can be used to build federated learning use cases based on existing machine learning projects.

<h2>Project Setup</h2>
Start by cloning the example project.

```bash
$ git clone https://github.com/i-anubhav-anand/Fed_learning_flower

```

containing files:
```bash

dataset.zip
   |__covid19
        |__ *.jpg (all image files)
   |__normal
         |__ *.jpg (all image files)
 --client.py
 --client2.py
 -- server.py
 -- README.md
 --requiremtns.txt
```
 <h1>From Centralized To Federated </h1>
 
  This Project is based on the Deep Learning with PyTorch that uses the custom data of Coivd and Normal Patient to predict if the patient is having covid or not.
  
  You can simply start the centralized training as described
  
  Set up Python 3 and new virtual environment
  
  Create virtual env.
  
  ```bash
    $ python -m venv venv
 ```
 Activate virtual env.
 ```bash
$ # Linux/macOS
$ source venv/bin/activate  
$ # Windows
$ venv\Scripts\activate 
 ```
 Installing all the dependencies
```bash
    $ pip install -r requirements.txt.
```
  Now that the server is running and waiting for clients, we can start two clients that will participate in the federated learning process.
  To do so simply open two more   terminal windows and run the following commands.
  
  Start the server in a terminal as follows:
  ```bash
    $ python server.py
```
  
  Start client 1 in the first terminal:

  ```bash
    $ python client.py
```

  Start client 2 in the second terminal:

```bash
    $ python client2.py
```  
  You are now training a PyTorch-based CNN image classifier on Chest X-ray(Custom Data), federated across two clients.
  
The models achieved a detection accuracy of COVID-19 around 65% trained on 200 images.
  

https://user-images.githubusercontent.com/76263415/173317577-2839e789-eb46-43ef-9e7d-c1b1156cfa23.mp4




  
