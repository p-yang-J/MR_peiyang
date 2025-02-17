using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Joint_2 : MonoBehaviour
{
    // Start is called before the first frame update
    public Joint_1 socketdat;
    void Start()
    {
        transform.position = new Vector3(-0.002f, -0.001149784f, 0.1074438f);
    }

    // Update is called once per frame
    private void Update()
    {
        Vector3 move_2 = socketdat.receivedPos;
        //float h = Input.GetAxis("Horizontal");
        //float v = Input.GetAxis("Vertical");
        //transform.Translate(new Vector3(v * Time.deltaTime * moveSpeed, 0, 0));
        float y = move_2.y;
        transform.localPosition = new Vector3(y, -0.001149784f, 0.1074438f);

    }
}
