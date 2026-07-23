#include "comms.h"
#include <stdio.h>

int main(void)
{
    Comms_Init(1);

    StatusReport report = {
        .system_id   = 1,
        .state       = STATE_ACTIVE,
        .uptime_sec  = 3600,
        .temperature = 25
    };

    int rc = Comms_SendStatus(&report);
    printf("SendStatus returned %d\n", rc);

    CommMessage rx;
    rc = Comms_Receive(&rx, 1000);
    printf("Receive returned %d, msg_id=0x%02X\n", rc, rx.msg_id);

    return 0;
}
