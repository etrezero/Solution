import socket

def get_hostname_from_ip(ip_address):
    try:
        hostname = socket.gethostbyaddr(ip_address)[0]
        return hostname
    except socket.herror:
        return "Hostname not found for the given IP address."

# Example usage
ip_address = "192.168.194.141"  # Replace with the desired IP address

IP_name = get_hostname_from_ip(ip_address)
print(f"Hostname: {IP_name}")
