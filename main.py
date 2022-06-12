from multiprocessing import Process
import client,client2
import server
import flwr as fl 
from flwr.server.strategy import FedAvg
import client
import time



def start_server(num_rounds: int, num_clients: int, fraction_fit: float):
    """Start the server with a slightly adjusted FedAvg strategy."""
    strategy = FedAvg(min_available_clients=num_clients, fraction_fit=fraction_fit)
    # Exposes the server by default on port 8080
    fl.server.start_server("localhost:8080",
          strategy=strategy,
        config={"num_rounds": num_rounds},
    )

def run():
    processes = []

        # Start the server
    server_process = Process(
            target=start_server, args=(1, 2, 0.5)
        )
    server_process.start()
    processes.append(server_process)

    time.sleep(10)

    print("Server started!")


    client_process = Process(target=client.main)
    client_process.start()
    processes.append(client_process)

    time.sleep(5)

    client_process2 = Process(target=client2.main)
    client_process2.start()
    processes.append(client_process2)

    for p in processes:
        p.join()


if __name__ == "__main__":   
    run()



