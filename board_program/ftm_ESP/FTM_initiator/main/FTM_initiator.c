#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_wifi.h"
#include "esp_mac.h"
#include "nvs_flash.h"
#include "sdkconfig.h"
#include <string.h>

const char *SSID = "JJ8428";

wifi_ftm_initiator_cfg_t ftmi_cfg = {
    .frm_count = 16,
    .burst_period = 2,
};

static const char *TAG_STA = "ftm_station";

static EventGroupHandle_t s_ftm_event_group;
static const int FTM_REPORT_BIT = BIT0;
static const int FTM_FAILURE_BIT = BIT1;
static wifi_ftm_report_entry_t *s_ftm_report;
static uint32_t s_rtt_est, s_dist_est; // , prev_s_rtt_est, prev_s_dist_est;

uint16_t g_scan_ap_num;
wifi_ap_record_t *g_ap_list_buffer;

EventBits_t bits;

static void event_handler(void *arg, esp_event_base_t event_base,
                          int32_t event_id, void *event_data)
{
    if (event_id == WIFI_EVENT_FTM_REPORT) {
        wifi_event_ftm_report_t *event = (wifi_event_ftm_report_t *) event_data;

        if (event->status == FTM_STATUS_SUCCESS) {
            s_rtt_est = event->rtt_est;
            s_dist_est = event->dist_est;
            s_ftm_report = event->ftm_report_data;
            xEventGroupSetBits(s_ftm_event_group, FTM_REPORT_BIT);
        } else {
            ESP_LOGI(TAG_STA, "FTM procedure with Peer("MACSTR") failed! (Status - %d)",
                MAC2STR(event->peer_mac), event->status);
            xEventGroupSetBits(s_ftm_event_group, FTM_FAILURE_BIT);
        }
    }
}

void initialize_wifi(void)
{
    // esp_log_level_set("wifi", ESP_LOG_WARN);
    esp_log_level_set("wifi", ESP_LOG_NONE);

    ESP_ERROR_CHECK(esp_netif_init());
    s_ftm_event_group = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id;
    ESP_ERROR_CHECK(
        esp_event_handler_instance_register(
            WIFI_EVENT,
            ESP_EVENT_ANY_ID,
            &event_handler,
            NULL,
            &instance_any_id
        )
    );

    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_NULL));
    ESP_ERROR_CHECK(esp_wifi_start());
}

wifi_ap_record_t *find_ftm_responder_ap()
{
    wifi_scan_config_t scan_config = { 0 };
    scan_config.ssid = (uint8_t *) SSID;

    ESP_ERROR_CHECK( esp_wifi_set_mode(WIFI_MODE_STA) );
    esp_wifi_scan_start(&scan_config, true);
    esp_wifi_scan_get_ap_num(&g_scan_ap_num);

    g_ap_list_buffer = malloc(g_scan_ap_num * sizeof(wifi_ap_record_t));
    esp_wifi_scan_get_ap_records(&g_scan_ap_num, (wifi_ap_record_t *)g_ap_list_buffer);
    for (int i = 0; i < g_scan_ap_num; i++) {
        if (strcmp((const char *)g_ap_list_buffer[i].ssid, SSID) == 0) {
            return &g_ap_list_buffer[i];
        }
    }
    free(g_ap_list_buffer);
    return NULL;
}


void app_main(void) 
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    initialize_wifi();

    wifi_ap_record_t *ap_record = find_ftm_responder_ap();
    if (ap_record == NULL) {
        ESP_LOGI(TAG_STA, "No FTM Responder with the SSID: %s", 
            SSID);
        return;
    } 

    memcpy(ftmi_cfg.resp_mac, ap_record->bssid, 6);
    ftmi_cfg.channel = ap_record->primary;
    
    while (true) {
        /*
        ESP_LOGI(TAG_STA, "Requesting FTM session with Frm Count - %d, Burst Period - %dmSec (0: No Preference)",
            ftmi_cfg.frm_count, ftmi_cfg.burst_period*100);
        ESP_LOGI(TAG_STA,"Free Heap Size: %zu bytes\n", 
            (size_t)esp_get_free_heap_size());
        */

        esp_wifi_ftm_initiate_session(&ftmi_cfg);

        bits = xEventGroupWaitBits(s_ftm_event_group, FTM_REPORT_BIT | FTM_FAILURE_BIT, pdTRUE, pdFALSE, portMAX_DELAY);
        if (bits & FTM_REPORT_BIT) {
            xEventGroupClearBits(s_ftm_event_group, FTM_REPORT_BIT);
            free(s_ftm_report);
            ESP_LOGI(TAG_STA, "{\"est_RTT\": %" PRId32 ", \"est_dist\": %" PRId32 ".%02" PRId32 ", \"ftm_fail\": false}",
                s_rtt_est, s_dist_est / 100, s_dist_est % 100);
        } else {
            xEventGroupClearBits(s_ftm_event_group, FTM_FAILURE_BIT);
            // ESP_LOGI(TAG_STA, "FTM Failed");
            ESP_LOGI(TAG_STA, "{\"est_RTT\": %" PRId32 ", \"est_dist\": %" PRId32 ".%02" PRId32 ", \"ftm_fail\": true}",
                s_rtt_est, s_dist_est / 100, s_dist_est % 100);
        }

        esp_wifi_ftm_end_session();
    }
}