#include "comms.h"
#include <string.h>

static uint8_t g_system_id = 0;
static uint8_t g_seq_counter = 0;

int Comms_Init(uint8_t system_id)
{
    g_system_id = system_id;
    g_seq_counter = 0;
    return 0;
}

uint16_t Comms_Checksum(const CommMessage *msg)
{
    uint16_t sum = 0;
    sum += msg->msg_id;
    sum += msg->seq_num;
    sum += msg->payload_len;
    for (uint16_t i = 0; i < msg->payload_len && i < MAX_PAYLOAD_LEN; i++) {
        sum += msg->payload[i];
    }
    return sum;
}

int Comms_Send(const CommMessage *msg)
{
    if (!msg || msg->payload_len > MAX_PAYLOAD_LEN) {
        return -1;
    }
    /* In real code this would write to a hardware bus */
    return 0;
}

int Comms_Receive(CommMessage *msg, uint32_t timeout_ms)
{
    if (!msg) {
        return -1;
    }
    (void)timeout_ms;
    memset(msg, 0, sizeof(CommMessage));
    return 0;
}

int Comms_SendStatus(const StatusReport *report)
{
    if (!report) {
        return -1;
    }

    CommMessage msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_id = MSG_ID_STATUS;
    msg.seq_num = g_seq_counter++;
    msg.payload_len = sizeof(StatusReport);
    memcpy(msg.payload, report, sizeof(StatusReport));
    msg.checksum = Comms_Checksum(&msg);

    return Comms_Send(&msg);
}
