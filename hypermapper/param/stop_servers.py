import grpc
from interopt.runner.grpc_runner import config_service_pb2 as cs
from interopt.runner.grpc_runner import config_service_pb2_grpc as cs_grpc

def send_shutdown_signal(server_address):
    with grpc.insecure_channel(f"{server_address}:50051") as channel:
        stub = cs_grpc.ConfigurationServiceStub(channel)
        request = cs.ShutdownRequest(shutdown=True)
        response = stub.Shutdown(request)
        if response.success:
            print(f"Shutdown signal sent successfully to {server_address}")
        else:
            print(f"Failed to send shutdown signal to {server_address}")

def main():
    server_addresses = [f"idun-10-{str(i).zfill(2)}" for i in range(3, 20)]
    for server in server_addresses:
        #print(server)
        send_shutdown_signal(server)
main()
