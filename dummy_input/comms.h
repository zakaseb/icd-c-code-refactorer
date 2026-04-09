#ifndef COMMS_H
#define COMMS_H

#include <stdint.h>

/* ICD v1.0 message definitions */

#define MSG_ID_HEARTBEAT   0x01
#define MSG_ID_STATUS      0x02
#define MSG_ID_COMMAND     0x03

#define MAX_PAYLOAD_LEN    64

typedef enum {
    STATE_IDLE    = 0,
    STATE_ACTIVE  = 1,
    STATE_ERROR   = 2
} SystemState;

typedef struct {
    uint8_t  msg_id;
    uint8_t  seq_num;
    uint16_t payload_len;
    uint8_t  payload[MAX_PAYLOAD_LEN];
    uint16_t checksum;
} CommMessage;

typedef struct {
    uint8_t     system_id;
    SystemState state;
    uint16_t    uptime_sec;
    int8_t      temperature;
} StatusReport;

/* Initialize the communication subsystem */
int  Comms_Init(uint8_t system_id);

/* Send a message over the bus */
int  Comms_Send(const CommMessage *msg);

/* Receive a message (blocking, timeout in ms) */
int  Comms_Receive(CommMessage *msg, uint32_t timeout_ms);

/* Build and send a status report */
int  Comms_SendStatus(const StatusReport *report);

/* Compute checksum for a message */
uint16_t Comms_Checksum(const CommMessage *msg);

#endif /* COMMS_H */
