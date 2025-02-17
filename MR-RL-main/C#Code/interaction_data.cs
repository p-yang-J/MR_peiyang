/*using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class interaction_data : MonoBehaviour
{
    public string serverIp = "127.0.0.1";
    public int serverPort = 8080;

    private UdpClient udpClient;

    void Start()
    {
        udpClient = new UdpClient();
    }

    void Update()
    {
        Vector3 handPosition = Input.mousePosition;
        string handPositionData = $"HandPosition: {handPosition.x}, {handPosition.y}, {handPosition.z}";
        byte[] handPositionBytes = Encoding.UTF8.GetBytes(handPositionData);
        udpClient.Send(handPositionBytes, handPositionBytes.Length, serverIp, serverPort);
        transform.localPosition = new Vector3(handPosition.x, handPosition.y, handPosition.z);
    }

    private void OnDestroy()
    {
        udpClient.Close();
    }
}
*/