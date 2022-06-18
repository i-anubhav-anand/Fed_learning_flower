# Federated_learning_flower


<h1>Introduction</h1>
How Flower can be used to build federated learning use cases based on existing machine learning projects.

<h2>Project Setup</h2>

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

 
  
  <h1>Hosting server.py on cloud(EC2) and adding port 8080 in security group.</h1>
  1.Log into the AWS Management Console.
  
  2.Scroll down the left navigation panel and choose "Security Group" under "Network & Security".
   
   ![image](https://user-images.githubusercontent.com/76263415/174425642-3f7d086a-1786-436b-a992-7ec4f1dbe087.png)
   
  3.Select the "EC2 Security Group" that needs to be verified.
  
  ![image](https://user-images.githubusercontent.com/76263415/174425658-56404bde-a5de-448d-bae1-c8ea8c47f9f2.png)

  4.Scroll down the bottom panel and choose "Inbound". Choose "Custom TCP rule" in the dropdown.Then you will be able to change the port to 8080.
   ![inbound](https://user-images.githubusercontent.com/76263415/174425457-bbc38b00-4534-47d8-b28d-0daf3e57958d.png)
  5.Click on the "Save" button to make the necessary changes.
  
  ![image](https://user-images.githubusercontent.com/76263415/174425684-675f2fab-6cc8-4049-94ac-61f033038975.png)
  
  
  Now, Connect to your EC2 instance.
  
  <h1>Clone the contents of the Repo into this directory using SSH.</h1>
  (Note: The ‘.’ at the end of the command is to put the contents of the repository into the current directory)
  
```bash

$ git clone https://github.com/i-anubhav-anand/Fed_learning_flower

```
     
   Go to the (Fed_learning_flower) folder and install all the dependencies

```bash
    $ pip install -r requirements.txt.
```
  After installation run 
  
```bash
    $ python3 server.py
```
  Now that the server is running and waiting for clients, we can start two clients that will participate in the federated learning process.
  To do so simply open two more   terminal windows and run the following commands.


 <h1>From Centralized To Federated </h1>
 
  This Project is based on the Deep Learning with PyTorch that uses the custom data of Coivd and Normal Patient to predict if the patient is having covid or not.
  
  <h1>Setting up to run client on your local system</h1>
  
  You can simply start the centralized training as described
  
  Start by cloning the example project.

```bash

$ git clone https://github.com/i-anubhav-anand/Fed_learning_flower

```
  
  
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
Before running client.py/client2.py make sure to replace localhost with the  public ip address of EC2 instance.
 
 ![image](https://user-images.githubusercontent.com/76263415/174426161-751c0702-bbd8-4fac-af7b-2d1e3f92f5d0.png)

which looks like this

![image](https://user-images.githubusercontent.com/76263415/174426355-ab394c55-fe69-4810-b3cf-18cdf784a7d0.png)



Start client 1 in the first terminal:

  ```bash
    $ python client.py
```

  Start client 2 in the second terminal:

```bash
    $ python client2.py
```  




  
  
  You are now training a PyTorch-based CNN image classifier on Chest X-ray(Custom Data), federated across two clients.
  


 <h1>Snapshot</h1>

![image](https://user-images.githubusercontent.com/76263415/174429467-d8d5c7d1-c6fc-4350-84d5-39e577d559d5.png)

<h1> Working Demo </h1>

https://user-images.githubusercontent.com/76263415/174444042-2a4c0aee-0da4-4132-9534-a8705c95785f.mp4





  
