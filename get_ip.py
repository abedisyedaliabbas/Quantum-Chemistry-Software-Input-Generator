#!/usr/bin/env python3
"""Quick script to get your local IP address for sharing the web app"""

import socket

def get_local_ip():
    """Get the local IP address"""
    try:
        # Connect to a remote address (doesn't actually send data)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

if __name__ == "__main__":
    ip = get_local_ip()
    print("=" * 60)
    print("Your local IP address is:", ip)
    print("=" * 60)
    print(f"\nShare this URL with your friend:")
    print(f"  http://{ip}:5000")
    print("\nMake sure:")
    print("  1. Your friend is on the same WiFi network")
    print("  2. The server is running (python quantum_steps_web.py)")
    print("  3. Firewall allows connections on port 5000")
    print("=" * 60)



