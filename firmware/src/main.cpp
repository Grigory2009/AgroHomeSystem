#include <Arduino.h>

#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ILI9341.h>
#include <XPT2046_Touchscreen.h>
#include <WiFi.h>
#include <Preferences.h>
#include <esp_arduino_version.h>
#include <math.h>
#include <WebServer.h>

// ==========================================
// АППАРАТНЫЕ ПИНЫ (ESP32-S3)
// ==========================================
#define TFT_MOSI  11
#define TFT_MISO  13
#define TFT_SCK   12
#define TFT_CS    10
#define TFT_DC    9
#define TFT_RST   14
#define TFT_BL    15    // Пин ШИМ-управления подсветкой экрана
#define PUMP_PIN  16    // Пин силового реле помпы полива

#define TOUCH_MOSI 6
#define TOUCH_MISO 5
#define TOUCH_SCK  7
#define TOUCH_CS   4

// Настройки ШИМ для подсветки
#define PWM_BL_CH    0
#define PWM_BL_FREQ  5000
#define PWM_BL_RES   8

// ==========================================
// ГЕОМЕТРИЯ ЭКРАНА И МАКЕТА
// ==========================================
#define SCREEN_W      320
#define SCREEN_H      240
#define HEADER_H      36
#define NAV_H         36
#define CARD_ROUND    8

// ==========================================
// ПАЛИТРЫ ОФОРМЛЕНИЯ (THEME ENGINE)
// ==========================================
#define RGB565(r, g, b) ((uint16_t)((((r) & 0xF8) << 8) | (((g) & 0xFC) << 3) | ((b) >> 3)))

enum ColorTheme {
  THEME_CYBER_EMERALD = 0,
  THEME_NORDIC_ICE    = 1,
  THEME_SOLAR_AMBER   = 2
};

struct ThemeColors {
  uint16_t bg;
  uint16_t surface;
  uint16_t surfaceHi;
  uint16_t border;
  uint16_t borderHi;
  uint16_t primary;
  uint16_t secondary;
  uint16_t accent;
  uint16_t ok;
  uint16_t warn;
  uint16_t alert;
  uint16_t txtMain;
  uint16_t txtMuted;
  uint16_t txtDim;
};

ColorTheme currentTheme = THEME_CYBER_EMERALD;
ThemeColors theme;

void applyTheme(ColorTheme t) {
  currentTheme = t;
  switch (t) {
    case THEME_CYBER_EMERALD:
      theme.bg         = RGB565(10, 14, 20);   // Deep Obsidian
      theme.surface    = RGB565(18, 25, 36);   // Dark Glass Surface
      theme.surfaceHi  = RGB565(28, 38, 54);   // Elevated Card
      theme.border     = RGB565(36, 50, 72);   // Subtle Frame
      theme.borderHi   = RGB565(56, 80, 114);  // Active Outline
      theme.primary    = RGB565(0, 245, 185);  // Neon Mint
      theme.secondary  = RGB565(56, 217, 240); // Electric Cyan
      theme.accent     = RGB565(140, 100, 255);// Cyber Violet
      theme.ok         = RGB565(38, 222, 129); // Vibrant Green
      theme.warn       = RGB565(255, 177, 66); // Amber
      theme.alert      = RGB565(255, 82, 82);  // Coral Red
      theme.txtMain    = RGB565(248, 250, 252);// Crisp White
      theme.txtMuted   = RGB565(140, 158, 182);// Secondary Slate
      theme.txtDim     = RGB565(82, 98, 122);  // Dim Labels
      break;

    case THEME_NORDIC_ICE:
      theme.bg         = RGB565(12, 16, 26);
      theme.surface    = RGB565(22, 30, 46);
      theme.surfaceHi  = RGB565(34, 46, 70);
      theme.border     = RGB565(44, 62, 92);
      theme.borderHi   = RGB565(70, 98, 140);
      theme.primary    = RGB565(64, 180, 255); // Glacier Blue
      theme.secondary  = RGB565(130, 224, 255);
      theme.accent     = RGB565(80, 240, 200);
      theme.ok         = RGB565(60, 220, 150);
      theme.warn       = RGB565(255, 185, 55);
      theme.alert      = RGB565(255, 90, 100);
      theme.txtMain    = RGB565(245, 250, 255);
      theme.txtMuted   = RGB565(145, 170, 200);
      theme.txtDim     = RGB565(85, 105, 135);
      break;

    case THEME_SOLAR_AMBER:
      theme.bg         = RGB565(16, 12, 10);
      theme.surface    = RGB565(30, 24, 18);
      theme.surfaceHi  = RGB565(48, 38, 28);
      theme.border     = RGB565(68, 52, 38);
      theme.borderHi   = RGB565(98, 76, 54);
      theme.primary    = RGB565(255, 175, 35); // Solar Gold
      theme.secondary  = RGB565(255, 215, 80);
      theme.accent     = RGB565(255, 105, 55);
      theme.ok         = RGB565(85, 225, 115);
      theme.warn       = RGB565(255, 175, 35);
      theme.alert      = RGB565(255, 75, 75);
      theme.txtMain    = RGB565(255, 250, 242);
      theme.txtMuted   = RGB565(185, 160, 135);
      theme.txtDim     = RGB565(115, 98, 82);
      break;
  }
}

// ==========================================
// ЛОКАЛИЗАЦИЯ И ВЫБОР ЯЗЫКА
// ==========================================
enum AppLanguage {
  LANG_EN = 0,
  LANG_RU = 1
};

AppLanguage currentLang = LANG_RU;

inline const char* tr(const char* en, const char* ru) {
  return (currentLang == LANG_RU) ? ru : en;
}

// ==========================================
// ШРИФТ 5x7 ДЛЯ КИРИЛЛИЦЫ UTF-8 (ILI9341)
// ==========================================
// 66 символов русского алфавита (33 заглавных + 33 строчных)
// Формат: 5 байт на символ (колонки 0..4, LSB вверху - bit0: строка 0 .. bit6: строка 6)
static const uint8_t cyrillic_glyphs[66][5] PROGMEM = {
  // Заглавные буквы (0..31: А..Я, 32: Ё)
  { 0x7E, 0x11, 0x11, 0x11, 0x7E }, // 0:  А
  { 0x7F, 0x49, 0x49, 0x49, 0x31 }, // 1:  Б
  { 0x7F, 0x49, 0x49, 0x49, 0x36 }, // 2:  В
  { 0x7F, 0x01, 0x01, 0x01, 0x01 }, // 3:  Г
  { 0x60, 0x3F, 0x21, 0x3F, 0x60 }, // 4:  Д
  { 0x7F, 0x49, 0x49, 0x49, 0x41 }, // 5:  Е
  { 0x49, 0x2A, 0x7F, 0x2A, 0x49 }, // 6:  Ж
  { 0x22, 0x41, 0x49, 0x49, 0x36 }, // 7:  З
  { 0x7F, 0x10, 0x08, 0x04, 0x7F }, // 8:  И
  { 0x7E, 0x13, 0x08, 0x07, 0x7E }, // 9:  Й
  { 0x7F, 0x08, 0x14, 0x22, 0x41 }, // 10: К
  { 0x60, 0x3E, 0x01, 0x7F, 0x40 }, // 11: Л
  { 0x7F, 0x02, 0x0C, 0x02, 0x7F }, // 12: М
  { 0x7F, 0x08, 0x08, 0x08, 0x7F }, // 13: Н
  { 0x3E, 0x41, 0x41, 0x41, 0x3E }, // 14: О
  { 0x7F, 0x01, 0x01, 0x01, 0x7F }, // 15: П
  { 0x7F, 0x09, 0x09, 0x09, 0x06 }, // 16: Р
  { 0x3E, 0x41, 0x41, 0x41, 0x22 }, // 17: С
  { 0x01, 0x01, 0x7F, 0x01, 0x01 }, // 18: Т
  { 0x03, 0x04, 0x78, 0x04, 0x03 }, // 19: У
  { 0x1C, 0x22, 0x7F, 0x22, 0x1C }, // 20: Ф
  { 0x63, 0x14, 0x08, 0x14, 0x63 }, // 21: Х
  { 0x7F, 0x40, 0x40, 0x7F, 0x60 }, // 22: Ц
  { 0x0F, 0x08, 0x08, 0x08, 0x7F }, // 23: Ч
  { 0x7F, 0x40, 0x7F, 0x40, 0x7F }, // 24: Ш
  { 0x7F, 0x40, 0x7F, 0x40, 0xFF }, // 25: Щ
  { 0x01, 0x7F, 0x48, 0x48, 0x30 }, // 26: Ъ
  { 0x7F, 0x48, 0x48, 0x30, 0x7F }, // 27: Ы
  { 0x7F, 0x48, 0x48, 0x48, 0x30 }, // 28: Ь
  { 0x22, 0x41, 0x49, 0x49, 0x3E }, // 29: Э
  { 0x7F, 0x08, 0x3E, 0x41, 0x3E }, // 30: Ю
  { 0x46, 0x29, 0x19, 0x09, 0x7F }, // 31: Я
  { 0x7E, 0x4B, 0x4A, 0x4B, 0x42 }, // 32: Ё

  // Строчные буквы (33..64: а..я, 65: ё)
  { 0x20, 0x54, 0x54, 0x54, 0x78 }, // 33: а
  { 0x3C, 0x4A, 0x49, 0x49, 0x30 }, // 34: б
  { 0x7C, 0x54, 0x54, 0x54, 0x28 }, // 35: в
  { 0x7C, 0x04, 0x04, 0x04, 0x04 }, // 36: г
  { 0x60, 0x3C, 0x24, 0x3C, 0x60 }, // 37: д
  { 0x38, 0x54, 0x54, 0x54, 0x18 }, // 38: е
  { 0x48, 0x28, 0x7C, 0x28, 0x48 }, // 39: ж
  { 0x24, 0x44, 0x44, 0x54, 0x28 }, // 40: з
  { 0x7C, 0x10, 0x08, 0x04, 0x7C }, // 41: и
  { 0x7C, 0x12, 0x09, 0x06, 0x7C }, // 42: й
  { 0x7C, 0x10, 0x28, 0x44, 0x00 }, // 43: к
  { 0x60, 0x38, 0x04, 0x7C, 0x40 }, // 44: л
  { 0x7C, 0x08, 0x10, 0x08, 0x7C }, // 45: м
  { 0x7C, 0x10, 0x10, 0x10, 0x7C }, // 46: н
  { 0x38, 0x44, 0x44, 0x44, 0x38 }, // 47: о
  { 0x7C, 0x04, 0x04, 0x04, 0x7C }, // 48: п
  { 0xFC, 0x24, 0x24, 0x24, 0x18 }, // 49: р
  { 0x38, 0x44, 0x44, 0x44, 0x20 }, // 50: с
  { 0x04, 0x04, 0x7C, 0x04, 0x04 }, // 51: т
  { 0x0C, 0x50, 0x50, 0x50, 0x3C }, // 52: у
  { 0x18, 0x24, 0x7E, 0x24, 0x18 }, // 53: ф
  { 0x44, 0x28, 0x10, 0x28, 0x44 }, // 54: х
  { 0x7C, 0x40, 0x40, 0x7C, 0x60 }, // 55: ц
  { 0x1C, 0x10, 0x10, 0x10, 0x7C }, // 56: ч
  { 0x7C, 0x40, 0x7C, 0x40, 0x7C }, // 57: ш
  { 0x7C, 0x40, 0x7C, 0x40, 0xFC }, // 58: щ
  { 0x04, 0x7C, 0x50, 0x50, 0x20 }, // 59: ъ
  { 0x7C, 0x50, 0x50, 0x20, 0x7C }, // 60: ы
  { 0x7C, 0x50, 0x50, 0x50, 0x20 }, // 61: ь
  { 0x28, 0x44, 0x54, 0x54, 0x38 }, // 62: э
  { 0x7C, 0x10, 0x38, 0x44, 0x38 }, // 63: ю
  { 0x48, 0x34, 0x14, 0x14, 0x7C }, // 64: я
  { 0x38, 0x56, 0x54, 0x56, 0x18 }  // 65: ё
};

// ==========================================
// КЛАСС ДИСПЛЕЯ С ПОДДЕРЖКОЙ КИРИЛЛИЦЫ UTF-8
// ==========================================
class AgroDisplay : public Adafruit_ILI9341 {
public:
  AgroDisplay(SPIClass *spiClass, int8_t dc, int8_t cs = -1, int8_t rst = -1)
    : Adafruit_ILI9341(spiClass, dc, cs, rst), utf8_lead(0) {}

  virtual size_t write(uint8_t c) override {
    if (utf8_lead == 0) {
      if (c == 0xD0 || c == 0xD1) {
        utf8_lead = c;
        return 1;
      }
      return Adafruit_ILI9341::write(c);
    }

    uint8_t b1 = utf8_lead;
    uint8_t b2 = c;
    utf8_lead = 0;

    int idx = -1;
    if (b1 == 0xD0) {
      if (b2 >= 0x90 && b2 <= 0xAF) {
        idx = b2 - 0x90; // 0..31: А..Я
      } else if (b2 == 0x81) {
        idx = 32;        // Ё
      } else if (b2 >= 0xB0 && b2 <= 0xBF) {
        idx = 33 + (b2 - 0xB0); // 33..48: а..п
      }
    } else if (b1 == 0xD1) {
      if (b2 >= 0x80 && b2 <= 0x8F) {
        idx = 49 + (b2 - 0x80); // 49..64: р..я
      } else if (b2 == 0x91) {
        idx = 65;        // ё
      }
    }

    if (idx >= 0 && idx < 66) {
      if (wrap && ((cursor_x + textsize_x * 6) > _width)) {
        cursor_x = 0;
        cursor_y += textsize_y * 8;
      }
      startWrite();
      for (int8_t i = 0; i < 5; i++) {
        uint8_t line = pgm_read_byte(&cyrillic_glyphs[idx][i]);
        for (int8_t j = 0; j < 8; j++, line >>= 1) {
          if (line & 1) {
            if (textsize_x == 1 && textsize_y == 1) {
              writePixel(cursor_x + i, cursor_y + j, textcolor);
            } else {
              writeFillRect(cursor_x + i * textsize_x, cursor_y + j * textsize_y, textsize_x, textsize_y, textcolor);
            }
          } else if (textbgcolor != textcolor) {
            if (textsize_x == 1 && textsize_y == 1) {
              writePixel(cursor_x + i, cursor_y + j, textbgcolor);
            } else {
              writeFillRect(cursor_x + i * textsize_x, cursor_y + j * textsize_y, textsize_x, textsize_y, textbgcolor);
            }
          }
        }
      }
      if (textbgcolor != textcolor) {
        if (textsize_x == 1 && textsize_y == 1) {
          writeFastVLine(cursor_x + 5, cursor_y, 8, textbgcolor);
        } else {
          writeFillRect(cursor_x + 5 * textsize_x, cursor_y, textsize_x, 8 * textsize_y, textbgcolor);
        }
      }
      endWrite();
      cursor_x += textsize_x * 6;
      return 1;
    }

    return Adafruit_ILI9341::write('?');
  }

private:
  uint8_t utf8_lead;
};

// Подсчет количества визуальных символов UTF-8 (для точного центрирования)
inline size_t utf8_char_count(const char* s) {
  if (!s) return 0;
  size_t count = 0;
  while (*s) {
    if ((*((const uint8_t*)s) & 0xC0) != 0x80) {
      count++;
    }
    s++;
  }
  return count;
}

SPIClass tftSPI = SPIClass(HSPI);
SPIClass touchSPI = SPIClass(FSPI);
AgroDisplay tft(&tftSPI, TFT_DC, TFT_CS, TFT_RST);
XPT2046_Touchscreen ts(TOUCH_CS);
Preferences prefs;

// ==========================================
// ДВИЖОК ЖЕСТОВ
// ==========================================
enum GestureType {
  GESTURE_NONE,
  GESTURE_TAP,
  GESTURE_SWIPE_UP,
  GESTURE_SWIPE_DOWN,
  GESTURE_SWIPE_LEFT,
  GESTURE_SWIPE_RIGHT
};

volatile bool newGestureAvailable = false;
volatile GestureType currentGesture = GESTURE_NONE;
volatile int gestureStartX = 0, gestureStartY = 0;
volatile int gestureEndX = 0, gestureEndY = 0;

// ==========================================
// ГЛОБАЛЬНОЕ СОСТОЯНИЕ
// ==========================================
int calX_min = 0, calX_max = 4095;
int calY_min = 0, calY_max = 4095;
bool swapAxes = false;
bool isCalibrated = false;

// Настройки подсветки и помпы
uint8_t brightnessLevel = 85; // 10..100%
bool pumpState = false;
bool pumpAutoMode = true;      // Режим автоматического таймера (15м ВКЛ / 45м ВЫКЛ)
unsigned long pumpTimerMark = 0;
unsigned long pumpManualOffAt = 0; // Таймер авто-отключения ручного полива
bool demoWavesEnabled = false; // Режим симуляции колебаний датчиков

// Показания датчиков
float current_pH        = 6.2;
int   current_TDS       = 820;
float current_waterTemp = 21.8;
float current_airTemp   = 23.5;
float current_humidity  = 54.0;
float current_VPD       = 1.05; // Дефицит упругости пара (VPD, кПа)

// Диапазоны нормы
const float PH_MIN_OK = 5.5, PH_MAX_OK = 6.8;
const int   TDS_MIN_OK = 650, TDS_MAX_OK = 950;
const float WT_MIN_OK = 18.0, WT_MAX_OK = 25.0;
const float AT_MIN_OK = 19.0, AT_MAX_OK = 27.0;
const float HUM_MIN_OK = 45.0, HUM_MAX_OK = 70.0;
const float VPD_MIN_OK = 0.8,  VPD_MAX_OK = 1.2;

// Типы метрик для детального экрана
enum MetricType {
  METRIC_PH = 0,
  METRIC_TDS = 1,
  METRIC_WATER_TEMP = 2,
  METRIC_AIR_TEMP = 3,
  METRIC_HUMIDITY = 4,
  METRIC_VPD = 5
};

enum TimeframePeriod {
  TF_1H = 0,
  TF_6H = 1,
  TF_24H = 2
};

MetricType currentDetailMetric = METRIC_PH;
TimeframePeriod currentDetailTF = TF_24H;

// Буферы истории для каждого показателя (24 отсчета)
#define HIST_POINTS 24
float metricHistory[6][HIST_POINTS];

// Кэш предыдущих значений для инкрементальной перерисовки
float lastDrawn_pH = -1;
int   lastDrawn_TDS = -1;
float lastDrawn_waterTemp = -1;
float lastDrawn_airTemp = -1;
float lastDrawn_humidity = -1;
float lastDrawn_VPD = -1;
int   lastDrawn_Health = -1;
bool  lastPumpState = false;
bool  lastWifiConnected = false;

String savedSSID = "";
String savedPass = "";

// Экраны
enum ScreenState { 
  SCR_HOME, 
  SCR_SPROUT, 
  SCR_GAME,
  SCR_LAMP,
  SCR_SETTINGS, 
  SCR_WIFI_SCAN, 
  SCR_KBD, 
  SCR_DETAIL, 
  SCR_ADVISOR,
  SCR_VISION
};
ScreenState currentScreen = SCR_HOME;

// ==========================================
// EDGE AI & RASPBERRY PI 4 СИНХРОНИЗАЦИЯ
// ==========================================
float  aiPlantHealth   = 96.0f; // Комплексный индекс здоровья (0..100%)
float  aiGrowthProgress = 35.0f; // Прогресс роста растения (0..100%)
int    aiGrowthStage   = 2;     // 1: Проросток, 2: Вегетация, 3: Цветение, 4: Зрелость
String aiStageName     = "VEGETATIVE";
float  aiBiomass       = 32.5f; // Площадь листвы ExG (%)
float  aiChlorosis     = 1.5f;  // Хлороз (%)
float  aiNecrosis      = 0.2f;  // Некроз (%)
String aiCropName      = "Tomato";
String aiDiagnosis     = "Healthy";
float  aiConfidence    = 98.5f; // Уверенность классификатора (%)
String aiSeverity      = "None";
String aiAdvice        = "Optimal foliage and biomass. Maintain current VPD.";
float  aiRpiTemp       = 45.0f; // Температура CPU Raspberry Pi
float  aiInferenceMs   = 18.2f; // Время инференса модели (мс)
unsigned long aiLastSyncMillis = 0;
bool   aiConnected     = false;
bool   aiHasEverSynced = false;

bool isAiConnected() {
  return (aiHasEverSynced && (millis() - aiLastSyncMillis < 15000UL));
}

// Локальный веб-сервер на порту 80
WebServer webServer(80);
bool webServerStarted = false;

// Таймер авто-засыпания экрана (60 секунд неактивности)
unsigned long lastTouchActivity = 0;
bool isSleeping = false;

// Wi-Fi список
String selectedSSID = "";
String inputPassword = "";
int wifiNetworkCount = 0;
int wifiOrder[16];
int wifiScrollOffset = 0;
                                                                                                                                                                                                                                                                                                                                            
enum KbdMode { KBD_LOWER, KBD_UPPER, KBD_NUM };                                                                                                                                                                                                                                                                                                                                            
KbdMode kbdMode = KBD_LOWER;                                                                                                                                                                                                                                                                                                                                            

struct KeyRect { int16_t x, y, w, h; String label; };
#define MAX_KEYS 40
KeyRect kbdKeys[MAX_KEYS];
uint8_t kbdKeyCount = 0;

const char* kbd_lower[4][10] = {
  {"q","w","e","r","t","y","u","i","o","p"},
  {"a","s","d","f","g","h","j","k","l","<-"},
  {"z","x","c","v","b","n","m",".","_","OK"},
  {"123","a/A","SPACE","SPACE","SPACE","SPACE","CLR","CLR","CLR","CLR"}
};
const char* kbd_upper[4][10] = {
  {"Q","W","E","R","T","Y","U","I","O","P"},
  {"A","S","D","F","G","H","J","K","L","<-"},
  {"Z","X","C","V","B","N","M",".","_","OK"},
  {"123","a/A","SPACE","SPACE","SPACE","SPACE","CLR","CLR","CLR","CLR"}
};
const char* kbd_num[4][10] = {
  {"1","2","3","4","5","6","7","8","9","0"},
  {"!","@","#","$","%","^","&","*","(",")"},
  {"-","+","=","/",":",";","?",",","<","<-"},
  {"abc","a/A","SPACE","SPACE","SPACE","SPACE","CLR","CLR","OK","OK"}
};

// ==========================================
// СОСТОЯНИЕ ТАМАГОЧИ (CYBER SPROUT)
// ==========================================
enum SproutMood {
  MOOD_HAPPY,
  MOOD_CHILL,
  MOOD_THIRSTY,
  MOOD_LOVE
};

SproutMood currentSproutMood = MOOD_HAPPY;
unsigned long sproutLoveUntil = 0;
int sproutLevel = 1;         // 1..4: Seedling -> Sprout -> Cyber-Bush -> Quantum Bloom
int sproutXP = 0;            // Текущие очки опыта
int sproutAgeDays = 5;
int currentWisdomIndex = 0;
int highScoreGame = 0;       // Личный рекорд в мини-игре

const char* agroWisdomQuotesEN[] = {
  "VPD is optimal! Stomata wide open.",
  "Photosynthesis at 99.4% peak rate!",
  "Clean roots, pure ions, top energy!",
  "Water at 22C gives my roots max O2.",
  "pH in sweet spot! Nutrients absorb.",
  "Mmm, fresh water stream feels great!",
  "Quantum LED spectrum is wonderful!",
  "Root ecosystem is 100% healthy!"
};

const char* agroWisdomQuotesRU[] = {
  "VPD в норме! Устьица открыты.",
  "Фотосинтез на пике: 99.4% мощности!",
  "Чистые корни, ионы, максимум сил!",
  "Вода 22C дает корням много O2.",
  "pH в точке оптимума! Питание идет.",
  "Ммм, свежий поток воды прекрасен!",
  "Спектр фитолампы идеален для роста!",
  "Корневая зона на 100% здорова!"
};
#define WISDOM_COUNT 8

// Состояние аркадной мини-игры (Hydro Catcher)
#define MAX_DROPS 4
struct GameDrop {
  float x, y, vy;
  int16_t prevX, prevY;
  uint8_t type; // 0: Water (+10), 1: Nutrient (+25), 2: Salt (-15)
  bool active;
};

GameDrop gameDrops[MAX_DROPS];
int gameScore = 0;
int basketX = 160;
int prevBasketX = 160;
unsigned long gameStartTime = 0;
bool gameActive = false;
bool gameOver = false;
unsigned long lastGameTick = 0;

// Состояние фито-светильника (Ambient Grow Lamp)
uint8_t lampColorIdx = 0;
uint8_t lampBrightnessIdx = 3;
const uint16_t lampColors[4] = { RGB565(255, 0, 30), RGB565(255, 140, 20), RGB565(40, 160, 255), RGB565(0, 255, 120) };
const char* lampNames[4] = { "BIO-RED 660nm", "SUNSET AMBER", "MOONLIGHT ICE", "AURORA GREEN" };
const uint8_t lampDuties[4] = { 25, 50, 75, 100 };

// 3D Starfield Screensaver Data
#define STAR_COUNT 42
struct Star3D {
  float x, y, z;
  int16_t prevSX, prevSY;
};
Star3D stars[STAR_COUNT];

TaskHandle_t LogicTask;

// Forward declarations
void drawHomeScreen();
void drawDetailScreen();
void drawSettingsScreen();
void drawAdvisorScreen();
void drawSproutScreen();
void drawWifiScanScreen();
void drawKeyboardScreen();
void refreshHomeValues();
void enterSleepMode();
void wakeFromSleepMode();
void updateStarfield();
void initStarfield();
void enterKeyboard(KbdMode mode);
void connectToWiFi();
void runCalibration();
void setupWebDashboard();
void startMiniGame();
void updateMiniGame();
void endMiniGame();
void exitMiniGame();
void drawLampScreen();
void exitLampMode();
void drawVisionScreen();
void processSerialCommunication();
void sendSerialTelemetry(const char* type);
void applyAiSyncData(const String& jsonStr);
void addSproutXP(int amount);
void saveSproutProgress();
void loadSproutProgress();
void resetSproutProgress();
void drawBottomTabs(int activeTab);
void drawSproutCharacter(int cx, int cy, int lvl, SproutMood mood);
void iconCrown(int cx, int cy, uint16_t color);
void animatePumpFlow();
void saveThemePreference();
void loadThemePreference();
void saveLanguagePreference();
void loadLanguagePreference();

// ==========================================
// УПРАВЛЕНИЕ ЯРКОСТЬЮ (HARDWARE PWM)
// ==========================================
void initBacklight() {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
  ledcAttach(TFT_BL, PWM_BL_FREQ, PWM_BL_RES);
#else
  ledcSetup(PWM_BL_CH, PWM_BL_FREQ, PWM_BL_RES);
  ledcAttachPin(TFT_BL, PWM_BL_CH);
#endif
}

void setDisplayBrightness(uint8_t percent) {
  percent = constrain(percent, 10, 100);
  brightnessLevel = percent;
  uint32_t duty = (percent * 255) / 100;
#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
  ledcWrite(TFT_BL, duty);
#else
  ledcWrite(PWM_BL_CH, duty);
#endif
  prefs.begin("display", false);
  prefs.putUChar("bright", brightnessLevel);
  prefs.end();
}

float calculateVPD(float airT, float hum) {
  float svp = 0.61078f * expf((17.27f * airT) / (airT + 237.3f));
  float avp = svp * (hum / 100.0f);
  float vpd = svp - avp;
  return (vpd < 0.0f) ? 0.0f : vpd;
}

int calculatePlantHealthScore() {
  int score = 100;
  if (current_pH < PH_MIN_OK || current_pH > PH_MAX_OK) score -= 22;
  if (current_TDS < TDS_MIN_OK || current_TDS > TDS_MAX_OK) score -= 22;
  if (current_waterTemp < WT_MIN_OK || current_waterTemp > WT_MAX_OK) score -= 14;
  if (current_airTemp < AT_MIN_OK || current_airTemp > AT_MAX_OK) score -= 14;
  if (current_humidity < HUM_MIN_OK || current_humidity > HUM_MAX_OK) score -= 14;
  if (current_VPD < VPD_MIN_OK || current_VPD > VPD_MAX_OK) score -= 14;
  return constrain(score, 15, 100);
}

void loadDisplaySettings() {
  prefs.begin("display", true);
  brightnessLevel = prefs.getUChar("bright", 85);
  prefs.end();
  setDisplayBrightness(brightnessLevel);
}

void saveThemePreference() {
  prefs.begin("uitheme", false);
  prefs.putUChar("theme", (uint8_t)currentTheme);
  prefs.end();
}

void loadThemePreference() {
  prefs.begin("uitheme", true);
  uint8_t t = prefs.getUChar("theme", 0);
  prefs.end();
  applyTheme((ColorTheme)t);
}

void saveLanguagePreference() {
  prefs.begin("locale", false);
  prefs.putUChar("lang", (uint8_t)currentLang);
  prefs.end();
}

void loadLanguagePreference() {
  prefs.begin("locale", true);
  uint8_t l = prefs.getUChar("lang", (uint8_t)LANG_RU);
  prefs.end();
  currentLang = (l <= 1) ? (AppLanguage)l : LANG_RU;
}

// ==========================================
// ЭНЕРГОНЕЗАВИСИМАЯ ПАМЯТЬ ДЛЯ WI-FI
// ==========================================
void saveWiFiCredentials(const String& ssid, const String& pass) {
  prefs.begin("wificreds", false);
  prefs.putString("ssid", ssid);
  prefs.putString("pass", pass);
  prefs.putBool("valid", true);
  prefs.end();
  savedSSID = ssid;
  savedPass = pass;
}

bool loadWiFiCredentials(String &ssid, String &pass) {
  prefs.begin("wificreds", true);
  bool valid = prefs.getBool("valid", false);
  if (valid) {
    ssid = prefs.getString("ssid", "");
    pass = prefs.getString("pass", "");
  }
  prefs.end();
  return valid && (ssid.length() > 0);
}

void forgetWiFiCredentials() {
  prefs.begin("wificreds", false);
  prefs.clear();
  prefs.end();
  savedSSID = "";
  savedPass = "";
  WiFi.disconnect(true);
}

// ==========================================
// МИНИМАЛИСТИЧНЫЕ ВЕКТОРНЫЕ ИКОНКИ
// ==========================================
void iconSignal(int x, int y, int level, uint16_t colorOn, uint16_t colorOff) {
  for (int i = 0; i < 4; i++) {
    int h = 3 + i * 3;
    uint16_t c = (i < level) ? colorOn : colorOff;
    tft.fillRect(x + i * 5, y + (12 - h), 3, h, c);
  }
}

void iconDot(int cx, int cy, int r, uint16_t color) {
  tft.fillCircle(cx, cy, r, color);
}

void iconChevronRight(int x, int y, int s, uint16_t color) {
  int half = s / 2;
  tft.drawLine(x, y, x + half, y + half, color);
  tft.drawLine(x + 1, y, x + half + 1, y + half, color);
  tft.drawLine(x + half, y + half, x, y + s, color);
  tft.drawLine(x + half + 1, y + half, x + 1, y + s, color);
}

void iconBackArrow(int x, int y, int s, uint16_t color) {
  int half = s / 2;
  tft.drawLine(x + half, y, x, y + half, color);
  tft.drawLine(x + half, y + 1, x, y + half + 1, color);
  tft.drawLine(x, y + half, x + half, y + s, color);
  tft.drawLine(x, y + half + 1, x + half, y + s + 1, color);
  tft.drawFastHLine(x, y + half, s, color);
}

void iconPower(int cx, int cy, int r, uint16_t color) {
  tft.drawCircle(cx, cy, r, color);
  tft.fillRect(cx - 2, cy - r - 1, 5, 5, theme.surface);
  tft.drawFastVLine(cx, cy - r, r, color);
}

void iconLeaf(int cx, int cy, uint16_t col) {
  tft.fillCircle(cx, cy, 4, col);
  tft.fillTriangle(cx - 3, cy, cx + 3, cy, cx, cy - 7, col);
}

void drawVectorHeart(int cx, int cy, int r, uint16_t color) {
  tft.fillCircle(cx - r / 2, cy - r / 3, r / 2, color);
  tft.fillCircle(cx + r / 2, cy - r / 3, r / 2, color);
  tft.fillTriangle(cx - r, cy - r / 4, cx + r, cy - r / 4, cx, cy + r, color);
}

void drawVectorDrop(int cx, int cy, int r, uint16_t color) {
  tft.fillCircle(cx, cy + 1, r, color);
  tft.fillTriangle(cx - r, cy + 1, cx + r, cy + 1, cx, cy - r - 2, color);
}

void iconCrown(int cx, int cy, uint16_t color) {
  tft.fillRect(cx - 10, cy + 2, 20, 3, color);
  tft.fillTriangle(cx - 10, cy + 2, cx - 10, cy - 6, cx - 5, cy + 2, color);
  tft.fillTriangle(cx - 5, cy + 2, cx, cy - 8, cx + 5, cy + 2, color);
  tft.fillTriangle(cx + 5, cy + 2, cx + 10, cy - 6, cx + 10, cy + 2, color);
  tft.fillCircle(cx - 10, cy - 7, 2, RGB565(255, 230, 80));
  tft.fillCircle(cx, cy - 9, 2, RGB565(255, 230, 80));
  tft.fillCircle(cx + 10, cy - 7, 2, RGB565(255, 230, 80));
}

// ==========================================
// ПРЕМИАЛЬНЫЕ КОМПОНЕНТЫ ИНТЕРФЕЙСА
// ==========================================
// Карточка со стильной неоновой световой полосой (Glow Accent Bar)
void drawGlowCard(int x, int y, int w, int h, uint16_t glowColor = 0, uint16_t bg = theme.surface, uint16_t border = theme.border) {
  tft.fillRoundRect(x, y, w, h, CARD_ROUND, bg);
  tft.drawRoundRect(x, y, w, h, CARD_ROUND, border);
  if (glowColor != 0) {
    // 2-пиксельная неоновая полоса по верхнему краю карточки
    tft.drawFastHLine(x + CARD_ROUND, y, w - 2 * CARD_ROUND, glowColor);
    tft.drawFastHLine(x + CARD_ROUND - 1, y + 1, w - 2 * CARD_ROUND + 2, glowColor);
  }
}

void drawCard(int x, int y, int w, int h, uint16_t bg = theme.surface, uint16_t border = theme.border) {
  drawGlowCard(x, y, w, h, 0, bg, border);
}

void drawPill(int x, int y, int w, int h, const char* text, uint16_t fg, uint16_t bg, uint16_t border = 0) {
  tft.fillRoundRect(x, y, w, h, h / 2, bg);
  if (border != 0) tft.drawRoundRect(x, y, w, h, h / 2, border);
  tft.setTextSize(1);
  tft.setTextColor(fg);
  int textLen = utf8_char_count(text) * 6;
  tft.setCursor(x + (w - textLen) / 2, y + (h - 8) / 2);
  tft.print(text);
}

void drawModernToggle(int x, int y, int w, int h, bool state) {
  uint16_t track = state ? theme.ok : theme.surfaceHi;
  uint16_t border = state ? theme.primary : theme.borderHi;
  tft.fillRoundRect(x, y, w, h, h / 2, track);
  tft.drawRoundRect(x, y, w, h, h / 2, border);

  int knobR = h / 2 - 3;
  int knobX = state ? (x + w - h / 2) : (x + h / 2);
  tft.fillCircle(knobX, y + h / 2, knobR, theme.txtMain);
}

// Миниатюрный трендовый график (Sparkline) внутри карточки
void drawSparkline(int x, int y, int w, int h, float *history, int count, uint16_t color) {
  if (count < 2) return;
  float minV = history[0], maxV = history[0];
  for (int i = 1; i < count; i++) {
    if (history[i] < minV) minV = history[i];
    if (history[i] > maxV) maxV = history[i];
  }
  float span = maxV - minV;
  if (span < 0.001f) span = 1.0f;

  int prevX = x;
  int prevY = y + h - (int)(((history[0] - minV) / span) * h);
  prevY = constrain(prevY, y, y + h);

  for (int i = 1; i < count; i++) {
    int curX = x + (i * w) / (count - 1);
    int curY = y + h - (int)(((history[i] - minV) / span) * h);
    curY = constrain(curY, y, y + h);
    tft.drawLine(prevX, prevY, curX, curY, color);
    prevX = curX;
    prevY = curY;
  }
  tft.fillCircle(prevX, prevY, 2, theme.txtMain);
}

// Автоматический перенос текста по словам внутри заданной ширины (с поддержкой UTF-8)
void drawWrappedText(const char* text, int x, int y, int maxW, int lineSpacing, int maxLines, uint16_t color) {
  tft.setTextSize(1);
  tft.setTextColor(color);
  int line = 0;
  int maxChars = maxW / 6;
  const char* p = text;

  while (*p && line < maxLines) {
    while (*p == ' ') p++;
    if (!*p) break;

    const char* lineStart = p;
    const char* lastSpace = nullptr;
    int charCount = 0;

    while (*p && *p != '\n') {
      if (*p == ' ') lastSpace = p;
      if ((*((const uint8_t*)p) & 0xC0) != 0x80) {
        if (charCount >= maxChars) {
          if (lastSpace && lastSpace > lineStart) {
            p = lastSpace;
          }
          break;
        }
        charCount++;
      }
      p++;
    }

    int byteLen = p - lineStart;
    char lineBuf[128];
    if (byteLen >= (int)sizeof(lineBuf)) byteLen = sizeof(lineBuf) - 1;
    memcpy(lineBuf, lineStart, byteLen);
    lineBuf[byteLen] = '\0';

    tft.setCursor(x, y + line * lineSpacing);
    tft.print(lineBuf);
    line++;

    if (*p == ' ') p++;
    else if (*p == '\n') p++;
  }
}

void buildKeyboardLayout(const char* kbd[4][10]) {
  kbdKeyCount = 0;
  const int KEY_W = 30, KEY_H = 40, KEY_GAP = 2;
  const int KBD_TOP = 46;

  for (int row = 0; row < 4; row++) {
    int col = 0;
    while (col < 10) {
      String label = String(kbd[row][col]);
      int span = 1;
      while (col + span < 10 && String(kbd[row][col + span]) == label) span++;
      int kx = 1 + col * (KEY_W + KEY_GAP);
      int ky = KBD_TOP + row * (KEY_H + KEY_GAP);
      int kw = span * KEY_W + (span - 1) * KEY_GAP;
      if (kbdKeyCount < MAX_KEYS) {
        kbdKeys[kbdKeyCount++] = { (int16_t)kx, (int16_t)ky, (int16_t)kw, (int16_t)KEY_H, label };
      }
      col += span;
    }
  }
}

uint16_t colorForKey(const String &label) {
  if (label == "OK") return theme.ok;
  if (label == "<-" || label == "CLR") return theme.alert;
  if (label == "SPACE") return theme.surface;
  if (label == "123" || label == "abc" || label == "a/A") return theme.secondary;
  return theme.surfaceHi;
}

// ==========================================
// КАЛИБРОВКА СЕНСОРНОГО СЛОЯ
// ==========================================
bool loadCalibration() {
  prefs.begin("touchcal", true);
  bool valid = prefs.getBool("valid", false);
  if (valid) {
    calX_min = prefs.getInt("xmin", 0);
    calX_max = prefs.getInt("xmax", 4095);
    calY_min = prefs.getInt("ymin", 0);
    calY_max = prefs.getInt("ymax", 4095);
    swapAxes = prefs.getBool("swap", false);
  }
  prefs.end();
  return valid;
}

void saveCalibration() {
  prefs.begin("touchcal", false);
  prefs.putBool("valid", true);
  prefs.putInt("xmin", calX_min);
  prefs.putInt("xmax", calX_max);
  prefs.putInt("ymin", calY_min);
  prefs.putInt("ymax", calY_max);
  prefs.putBool("swap", swapAxes);
  prefs.end();
}

void drawCalibrationPoint(int x, int y, const char* prompt) {
  tft.fillScreen(theme.bg);
  tft.drawFastHLine(x - 10, y, 21, theme.primary);
  tft.drawFastVLine(x, y - 10, 21, theme.primary);
  tft.drawCircle(x, y, 5, theme.txtMain);
  tft.setTextSize(1);
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(30, 210);
  tft.print(prompt);
}

void getRawTouch(int &rawX, int &rawY) {
  long avgX = 0, avgY = 0;
  int samples = 16;
  for (int i = 0; i < samples; i++) {
    TS_Point p = ts.getPoint();
    avgX += p.x; avgY += p.y;
    delay(4);
  }
  rawX = avgX / samples;
  rawY = avgY / samples;
}

void runCalibration() {
  int p1_x, p1_y, p2_x, p2_y, p3_x, p3_y;
  drawCalibrationPoint(30, 30, "Touch crosshair 1/3 (Top Left)");
  while (!ts.touched()) delay(10);
  getRawTouch(p1_x, p1_y);
  while (ts.touched()) delay(10);

  drawCalibrationPoint(290, 30, "Touch crosshair 2/3 (Top Right)");
  while (!ts.touched()) delay(10);
  getRawTouch(p2_x, p2_y);
  while (ts.touched()) delay(10);

  drawCalibrationPoint(30, 210, "Touch crosshair 3/3 (Bottom Left)");
  while (!ts.touched()) delay(10);
  getRawTouch(p3_x, p3_y);
  while (ts.touched()) delay(10);

  int deltaX = abs(p2_x - p1_x);
  int deltaY = abs(p2_y - p1_y);
  if (deltaY > deltaX) {
    swapAxes = true;
    calX_min = p1_y; calX_max = p2_y;
    calY_min = p1_x; calY_max = p3_x;
  } else {
    swapAxes = false;
    calX_min = p1_x; calX_max = p2_x;
    calY_min = p1_y; calY_max = p3_y;
  }
  isCalibrated = true;
  saveCalibration();
  tft.fillScreen(theme.bg);
}

// ==========================================
// ФОНОВЫЙ ТАСК КАСАНИЙ И ЖЕСТОВ
// ==========================================
void TaskCore1(void * pvParameters) {
  bool isTouching = false;
  int startX = 0, startY = 0;
  int lastX = 0, lastY = 0;
  unsigned long pressTime = 0;

  for (;;) {
    if (isCalibrated) {
      if (ts.touched()) {
        TS_Point p = ts.getPoint();
        int calculatedX, calculatedY;
        if (swapAxes) {
          calculatedX = map(p.y, calX_min, calX_max, 30, 290);
          calculatedY = map(p.x, calY_min, calY_max, 30, 210);
        } else {
          calculatedX = map(p.x, calX_min, calX_max, 30, 290);
          calculatedY = map(p.y, calY_min, calY_max, 30, 210);
        }
        int curX = constrain(calculatedX, 0, SCREEN_W);
        int curY = constrain(calculatedY, 0, SCREEN_H);

        if (!isTouching) {
          isTouching = true;
          startX = curX;
          startY = curY;
          pressTime = millis();
        }
        lastX = curX;
        lastY = curY;
      } else {
        if (isTouching) {
          int dx = lastX - startX;
          int dy = lastY - startY;
          unsigned long duration = millis() - pressTime;

          const int SWIPE_DIST = 32;

          if (duration < 550) {
            if (abs(dx) > SWIPE_DIST && abs(dx) > abs(dy) * 1.3) {
              currentGesture = (dx > 0) ? GESTURE_SWIPE_RIGHT : GESTURE_SWIPE_LEFT;
            } else if (abs(dy) > SWIPE_DIST && abs(dy) > abs(dx) * 1.3) {
              currentGesture = (dy > 0) ? GESTURE_SWIPE_DOWN : GESTURE_SWIPE_UP;
            } else {
              currentGesture = GESTURE_TAP;
            }
          } else {
            currentGesture = GESTURE_TAP;
          }

          gestureStartX = startX;
          gestureStartY = startY;
          gestureEndX = lastX;
          gestureEndY = lastY;
          newGestureAvailable = true;
          isTouching = false;
        }
      }
    }
    vTaskDelay(15 / portTICK_PERIOD_MS);
  }
}

// ==========================================
// ШАПКА И СИСТЕМНЫЙ HUD
// ==========================================
String getFormattedUptime() {
  unsigned long s = millis() / 1000;
  int hrs = s / 3600;
  int mins = (s % 3600) / 60;
  int secs = s % 60;
  char buf[12];
  snprintf(buf, sizeof(buf), "%02d:%02d:%02d", hrs, mins, secs);
  return String(buf);
}

void drawMinimalHeader(const char* title, bool showBack = false) {
  tft.fillRect(0, 0, SCREEN_W, HEADER_H, theme.bg);
  tft.drawFastHLine(0, HEADER_H - 1, SCREEN_W, theme.border);

  if (showBack) {
    iconBackArrow(12, 11, 14, theme.primary);
    tft.setTextSize(2);
    tft.setTextColor(theme.txtMain);
    tft.setCursor(38, 10);
    tft.print(title);
  } else {
    // Логотип с акцентной точкой
    iconDot(14, 18, 4, theme.primary);
    tft.setTextSize(2);
    tft.setTextColor(theme.txtMain);
    tft.setCursor(24, 10);
    tft.print(title);

    // Индикатор Plant Health Score (тап открывает Edge AI Vision / Советник)
    bool aiActive = isAiConnected();
    int health = aiActive ? (int)aiPlantHealth : calculatePlantHealthScore();
    uint16_t badgeColor = (health >= 85) ? theme.ok : (health >= 65 ? theme.warn : theme.alert);
    const char* hStatus = (health >= 85 ? tr("PRIME", "НОРМА") : (health >= 65 ? tr("FAIR", "ВНИМ") : tr("ATTN", "СБОЙ")));
    String healthStr = (aiActive ? "AI " : "") + String(health) + "% " + hStatus;
    drawPill(118, 8, 92, 20, healthStr.c_str(), badgeColor, theme.surfaceHi, aiActive ? theme.primary : theme.border);

    // Uptime & CPU Temp HUD
    int cpuTemp = (int)temperatureRead();
    tft.setTextSize(1);
    tft.setTextColor(theme.txtDim);
    tft.setCursor(216, 7);
    tft.print(String(cpuTemp) + "C");

    tft.setCursor(216, 19);
    tft.print(getFormattedUptime().substring(3)); // mm:ss

    // Статус Wi-Fi и яркости в правом углу
    bool connected = (WiFi.status() == WL_CONNECTED);
    iconSignal(270, 12, connected ? 4 : 1, connected ? theme.ok : theme.alert, theme.borderHi);

    tft.setTextColor(theme.txtMuted);
    tft.setCursor(295, 14);
    tft.print(String(brightnessLevel));
  }
}

// 4-вкладочная навигационная панель: [MONITOR/МОНИТОР] [AI VISION/ИИ ЗРЕНИЕ] [SPROUT/РОСТОК] [SETTINGS/НАСТР.]
void drawBottomTabs(int activeTab) {
  int tabY = SCREEN_H - NAV_H;
  tft.fillRect(0, tabY, SCREEN_W, NAV_H, theme.bg);
  tft.drawFastHLine(0, tabY, SCREEN_W, theme.border);
  tft.setTextSize(1);

  // Таб 0: MONITOR / МОНИТОР (x: 0..79, center = 40)
  bool isMon = (activeTab == 0);
  uint16_t colMon = isMon ? theme.primary : theme.txtDim;
  const char* txtMon = tr("MONITOR", "МОНИТОР");
  int wMon = utf8_char_count(txtMon) * 6;
  tft.setTextColor(colMon);
  tft.setCursor(40 - wMon / 2, tabY + 14);
  tft.print(txtMon);
  if (isMon) tft.fillRoundRect(40 - (wMon + 8) / 2, tabY + 28, wMon + 8, 3, 1, theme.primary);

  // Таб 1: AI VISION / ИИ ЗРЕНИЕ (x: 80..159, center = 120)
  bool isAi = (activeTab == 1);
  uint16_t colAi = isAi ? theme.primary : theme.txtDim;
  const char* txtAi = tr("AI VISION", "ИИ ЗРЕНИЕ");
  int wAi = utf8_char_count(txtAi) * 6;
  tft.setTextColor(colAi);
  tft.setCursor(120 - wAi / 2, tabY + 14);
  tft.print(txtAi);
  if (isAi) tft.fillRoundRect(120 - (wAi + 8) / 2, tabY + 28, wAi + 8, 3, 1, theme.primary);

  // Таб 2: SPROUT / РОСТОК (x: 160..239, center = 200)
  bool isSprout = (activeTab == 2);
  uint16_t colSprout = isSprout ? theme.primary : theme.txtDim;
  const char* txtSprout = tr("SPROUT", "РОСТОК");
  int wSprout = utf8_char_count(txtSprout) * 6;
  tft.setTextColor(colSprout);
  tft.setCursor(200 - wSprout / 2, tabY + 14);
  tft.print(txtSprout);
  if (isSprout) tft.fillRoundRect(200 - (wSprout + 8) / 2, tabY + 28, wSprout + 8, 3, 1, theme.primary);

  // Таб 3: SETTINGS / НАСТРОЙКИ (x: 240..319, center = 280)
  bool isSet = (activeTab == 3);
  uint16_t colSet = isSet ? theme.primary : theme.txtDim;
  const char* txtSet = tr("SETTINGS", "НАСТР.");
  int wSet = utf8_char_count(txtSet) * 6;
  tft.setTextColor(colSet);
  tft.setCursor(280 - wSet / 2, tabY + 14);
  tft.print(txtSet);
  if (isSet) tft.fillRoundRect(280 - (wSet + 8) / 2, tabY + 28, wSet + 8, 3, 1, theme.primary);
}

// ==========================================
// АНИМИРОВАННЫЙ КИНЕТИЧЕСКИЙ ИНДИКАТОР ПОТОКА ПОМПЫ
// ==========================================
void animatePumpFlow() {
  if (!pumpState || currentScreen != SCR_HOME || isSleeping) return;
  static unsigned long lastFlowTick = 0;
  if (millis() - lastFlowTick < 80) return;
  lastFlowTick = millis();

  static int flowOffset = 0;
  flowOffset = (flowOffset + 3) % 20;

  int fx = 104, fy = 180, fw = 132, fh = 7;
  tft.fillRoundRect(fx, fy, fw, fh, 3, theme.surfaceHi);

  for (int i = -20 + flowOffset; i < fw; i += 20) {
    int start = max(fx, fx + i);
    int end = min(fx + fw, fx + i + 10);
    if (end > start) {
      tft.fillRoundRect(start, fy + 1, end - start, fh - 2, 2, theme.primary);
    }
  }
}

// ==========================================
// ДЕТАЛЬНЫЙ ЭКРАН И ГРАФИК ПОКАЗАТЕЛЯ
// ==========================================
void getMetricMeta(MetricType type, String &title, String &unit, float &val, float &minOk, float &maxOk, int &decimals) {
  switch (type) {
    case METRIC_PH:
      title = tr("pH LEVEL", "УРОВЕНЬ pH"); unit = "pH"; val = current_pH;
      minOk = PH_MIN_OK; maxOk = PH_MAX_OK; decimals = 1;
      break;
    case METRIC_TDS:
      title = tr("NUTRIENTS (TDS)", "ПИТАНИЕ (TDS)"); unit = "ppm"; val = (float)current_TDS;
      minOk = TDS_MIN_OK; maxOk = TDS_MAX_OK; decimals = 0;
      break;
    case METRIC_WATER_TEMP:
      title = tr("WATER TEMP", "ТЕМП. ВОДЫ"); unit = "C"; val = current_waterTemp;
      minOk = WT_MIN_OK; maxOk = WT_MAX_OK; decimals = 1;
      break;
    case METRIC_AIR_TEMP:
      title = tr("AIR TEMP", "ТЕМП. ВОЗДУХА"); unit = "C"; val = current_airTemp;
      minOk = AT_MIN_OK; maxOk = AT_MAX_OK; decimals = 1;
      break;
    case METRIC_HUMIDITY:
      title = tr("HUMIDITY", "ВЛАЖНОСТЬ"); unit = "%"; val = current_humidity;
      minOk = HUM_MIN_OK; maxOk = HUM_MAX_OK; decimals = 0;
      break;
    case METRIC_VPD:
      title = tr("VPD (DEFICIT)", "VPD (ДЕФИЦИТ)"); unit = "kPa"; val = current_VPD;
      minOk = VPD_MIN_OK; maxOk = VPD_MAX_OK; decimals = 2;
      break;
  }
}

void drawDetailScreen() {
  tft.fillScreen(theme.bg);

  String title, unit;
  float currentVal, minOk, maxOk;
  int decimals;
  getMetricMeta(currentDetailMetric, title, unit, currentVal, minOk, maxOk, decimals);

  bool isOk = (currentVal >= minOk && currentVal <= maxOk);

  // 1. Шапка с кнопкой назад и статусом
  tft.fillRect(0, 0, SCREEN_W, HEADER_H, theme.bg);
  tft.drawFastHLine(0, HEADER_H - 1, SCREEN_W, theme.border);
  iconBackArrow(12, 11, 14, theme.primary);

  tft.setTextSize(2);
  tft.setTextColor(theme.txtMain);
  tft.setCursor(38, 10);
  tft.print(title);

  drawPill(234, 8, 78, 20, isOk ? tr("OPTIMAL", "НОРМА") : tr("ATTENTION", "ВНИМАНИЕ"), isOk ? theme.ok : theme.warn, theme.surfaceHi);

  // 2. Статистика (Min, Max, Avg) по истории
  float minVal = 99999.0f, maxVal = -99999.0f, sumVal = 0.0f;
  for (int i = 0; i < HIST_POINTS; i++) {
    float v = metricHistory[currentDetailMetric][i];
    if (v < minVal) minVal = v;
    if (v > maxVal) maxVal = v;
    sumVal += v;
  }
  float avgVal = sumVal / (float)HIST_POINTS;

  // 3. Блок текущего значения и статистики со светящейся линией
  drawGlowCard(8, 42, 304, 46, theme.primary, theme.surface, theme.border);

  // Большое значение слева
  tft.setTextSize(3);
  tft.setTextColor(isOk ? theme.txtMain : theme.warn);
  tft.setCursor(18, 52);
  tft.print(String(currentVal, decimals));

  tft.setTextSize(1);
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(tft.getCursorX() + 4, 66);
  tft.print(unit);

  tft.drawFastVLine(124, 48, 34, theme.border);

  tft.setTextSize(1);
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(134, 48);
  tft.print("MIN: ");
  tft.setTextColor(theme.txtMain);
  tft.print(String(minVal, decimals));

  tft.setTextColor(theme.txtMuted);
  tft.setCursor(224, 48);
  tft.print("MAX: ");
  tft.setTextColor(theme.txtMain);
  tft.print(String(maxVal, decimals));

  tft.setTextColor(theme.txtMuted);
  tft.setCursor(134, 68);
  tft.print("AVG: ");
  tft.setTextColor(theme.txtMain);
  tft.print(String(avgVal, decimals));

  tft.setTextColor(theme.txtMuted);
  tft.setCursor(224, 68);
  tft.print("TGT: ");
  tft.setTextColor(theme.primary);
  tft.print(String(minOk, 0) + "-" + String(maxOk, 0));

  // 4. Селектор интервалов (1H, 6H, 24H)
  int tfY = 94;
  tft.setTextSize(1);
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(12, tfY + 5);
  tft.print(tr("TIMEFRAME:", "ПЕРИОД:"));

  const char* tfLabels[3] = { "1H", "6H", "24H" };
  for (int i = 0; i < 3; i++) {
    int btnX = 176 + i * 46;
    bool active = (currentDetailTF == (TimeframePeriod)i);
    drawPill(btnX, tfY, 40, 18, tfLabels[i], active ? theme.bg : theme.txtMuted, active ? theme.primary : theme.surfaceHi);
  }

  // 5. График в отдельной карточке
  const int cX = 8, cY = 118, cW = 304, cH = 114;
  drawCard(cX, cY, cW, cH, theme.surface, theme.border);

  const int pX = cX + 38;
  const int pY = cY + 12;
  const int pW = cW - 48; // 256 px
  const int pH = cH - 32; // 82 px

  float plotMin = min(minVal, minOk) * 0.95f;
  float plotMax = max(maxVal, maxOk) * 1.05f;
  if (plotMax - plotMin < 0.2f) plotMax = plotMin + 1.0f;

  int yNormTop = pY + pH - (int)(((maxOk - plotMin) / (plotMax - plotMin)) * pH);
  int yNormBot = pY + pH - (int)(((minOk - plotMin) / (plotMax - plotMin)) * pH);
  yNormTop = constrain(yNormTop, pY, pY + pH);
  yNormBot = constrain(yNormBot, pY, pY + pH);

  if (yNormBot > yNormTop) {
    tft.fillRect(pX, yNormTop, pW, yNormBot - yNormTop, RGB565(16, 32, 34));
    tft.drawFastHLine(pX, yNormTop, pW, RGB565(26, 68, 64));
    tft.drawFastHLine(pX, yNormBot, pW, RGB565(26, 68, 64));
  }

  tft.setTextSize(1);
  tft.setTextColor(theme.txtDim);
  tft.drawFastHLine(pX, pY, pW, theme.border);
  tft.setCursor(cX + 6, pY - 3);
  tft.print(String(plotMax, decimals));

  int midY = pY + pH / 2;
  tft.drawFastHLine(pX, midY, pW, theme.border);
  tft.setCursor(cX + 6, midY - 3);
  tft.print(String((plotMax + plotMin) / 2.0f, decimals));

  tft.drawFastHLine(pX, pY + pH, pW, theme.border);
  tft.setCursor(cX + 6, pY + pH - 4);
  tft.print(String(plotMin, decimals));

  // Отрисовка линии тренда графика
  int prevPtX = 0, prevPtY = 0;
  for (int i = 0; i < HIST_POINTS; i++) {
    float v = metricHistory[currentDetailMetric][i];
    int ptX = pX + (i * pW) / (HIST_POINTS - 1);
    int ptY = pY + pH - (int)(((v - plotMin) / (plotMax - plotMin)) * pH);
    ptY = constrain(ptY, pY, pY + pH);

    if (i > 0) {
      tft.drawLine(prevPtX, prevPtY, ptX, ptY, theme.primary);
      tft.drawLine(prevPtX, prevPtY + 1, ptX, ptY + 1, theme.primary);
    }
    tft.fillCircle(ptX, ptY, 2, theme.txtMain);

    prevPtX = ptX;
    prevPtY = ptY;
  }

  // Временные метки оси X
  tft.setTextSize(1);
  tft.setTextColor(theme.txtDim);
  const char* tStart = (currentDetailTF == TF_1H) ? "-60m" : (currentDetailTF == TF_6H) ? "-6h" : "-24h";
  const char* tMid   = (currentDetailTF == TF_1H) ? "-30m" : (currentDetailTF == TF_6H) ? "-3h" : "-12h";
  tft.setCursor(pX, pY + pH + 6);
  tft.print(tStart);
  tft.setCursor(pX + pW / 2 - 12, pY + pH + 6);
  tft.print(tMid);
  tft.setCursor(pX + pW - 20, pY + pH + 6);
  tft.print(tr("Now", "Сейч"));
}

void appendMetricSample(MetricType type, float val) {
  for (int i = 0; i < HIST_POINTS - 1; i++) {
    metricHistory[type][i] = metricHistory[type][i + 1];
  }
  metricHistory[type][HIST_POINTS - 1] = val;
}

void initMetricHistory() {
  for (int i = 0; i < HIST_POINTS; i++) {
    float progress = (float)i / (float)HIST_POINTS;
    float wave = sinf(progress * 6.28f);

    metricHistory[METRIC_PH][i]         = current_pH + wave * 0.25f + ((i % 3) - 1) * 0.05f;
    metricHistory[METRIC_TDS][i]        = (float)current_TDS + wave * 45.0f + ((i % 4) - 2) * 8.0f;
    metricHistory[METRIC_WATER_TEMP][i] = current_waterTemp + wave * 0.8f;
    metricHistory[METRIC_AIR_TEMP][i]   = current_airTemp + wave * 1.5f;
    metricHistory[METRIC_HUMIDITY][i]   = current_humidity + wave * 4.0f;
    metricHistory[METRIC_VPD][i]        = current_VPD + wave * 0.15f;
  }
}

// ==========================================
// ЛОГИКА ПРОКАЧКИ И ЭВОЛЮЦИИ ТАМАГОЧИ
// ==========================================
void saveSproutProgress() {
  prefs.begin("sproutprog", false);
  prefs.putInt("lvl", sproutLevel);
  prefs.putInt("xp", sproutXP);
  prefs.putInt("hscore", highScoreGame);
  prefs.end();
}

void loadSproutProgress() {
  prefs.begin("sproutprog", true);
  sproutLevel = prefs.getInt("lvl", 1);
  sproutXP = prefs.getInt("xp", 0);
  highScoreGame = prefs.getInt("hscore", 0);
  prefs.end();
  if (sproutLevel < 1) sproutLevel = 1;
  if (sproutLevel > 4) sproutLevel = 4;
}

void addSproutXP(int amount) {
  sproutXP += amount;
  int neededXP = sproutLevel * 100;
  if (sproutXP >= neededXP) {
    if (sproutLevel < 4) {
      sproutXP -= neededXP;
      sproutLevel++;
      sproutLoveUntil = millis() + 8000; // Праздничный восторг от эволюции
    } else {
      sproutXP = neededXP; // Максимальный уровень
    }
  }
  saveSproutProgress();
}

void resetSproutProgress() {
  sproutLevel = 1;
  sproutXP = 0;
  saveSproutProgress();
  aiGrowthStage = 1;
  aiGrowthProgress = 5.0f;
  sproutLoveUntil = millis() + 4000;
  Serial.println("{\"type\":\"cmd\",\"action\":\"reset_sprout\"}");
}

// 4 стадии эволюции персонажа-ростка
void drawSproutCharacter(int cx, int cy, int lvl, SproutMood mood) {
  // 1. Горшок гидропоники
  if (lvl == 1) {
    tft.fillRoundRect(cx - 22, cy + 14, 44, 20, 5, theme.surfaceHi);
    tft.drawRoundRect(cx - 22, cy + 14, 44, 20, 5, theme.borderHi);
    tft.fillRoundRect(cx - 16, cy + 24, 32, 4, 2, theme.secondary);
  } else if (lvl == 2) {
    tft.fillRoundRect(cx - 30, cy + 10, 60, 26, 6, theme.surfaceHi);
    tft.drawRoundRect(cx - 30, cy + 10, 60, 26, 6, theme.borderHi);
    tft.fillRoundRect(cx - 24, cy + 24, 48, 6, 2, theme.secondary);
  } else if (lvl == 3) {
    tft.fillRoundRect(cx - 34, cy + 8, 68, 28, 6, theme.surfaceHi);
    tft.drawRoundRect(cx - 34, cy + 8, 68, 28, 6, theme.primary);
    tft.fillRoundRect(cx - 28, cy + 16, 56, 4, 2, theme.primary);
    tft.fillRoundRect(cx - 28, cy + 24, 56, 4, 2, theme.secondary);
  } else { // lvl >= 4
    tft.fillRoundRect(cx - 36, cy + 6, 72, 30, 7, RGB565(40, 32, 20));
    tft.drawRoundRect(cx - 36, cy + 6, 72, 30, 7, RGB565(255, 200, 50));
    tft.fillRoundRect(cx - 30, cy + 14, 60, 5, 2, RGB565(255, 200, 50));
    tft.fillRoundRect(cx - 30, cy + 23, 60, 5, 2, theme.primary);
  }

  // 2. Стебель и листья
  if (lvl == 1) {
    tft.fillRect(cx - 2, cy - 16, 4, 30, theme.ok);
    tft.fillCircle(cx - 10, cy - 8, 6, theme.ok);
    tft.fillCircle(cx + 10, cy - 8, 6, theme.ok);
    tft.fillCircle(cx, cy - 24, 14, theme.ok);
  } else if (lvl == 2) {
    tft.fillRect(cx - 2, cy - 25, 4, 35, theme.ok);
    tft.fillCircle(cx - 16, cy - 14, 9, theme.ok);
    tft.fillCircle(cx + 16, cy - 14, 9, theme.ok);
    tft.fillCircle(cx - 10, cy - 26, 7, theme.primary);
    tft.fillCircle(cx + 10, cy - 26, 7, theme.primary);
    tft.fillCircle(cx, cy - 35, 18, theme.ok);
  } else if (lvl == 3) {
    tft.fillRect(cx - 3, cy - 30, 6, 38, theme.ok);
    tft.fillCircle(cx - 20, cy - 10, 10, theme.ok);
    tft.fillCircle(cx + 20, cy - 10, 10, theme.ok);
    tft.fillCircle(cx - 18, cy - 24, 8, theme.primary);
    tft.fillCircle(cx + 18, cy - 24, 8, theme.primary);
    tft.fillCircle(cx - 10, cy - 34, 7, theme.secondary);
    tft.fillCircle(cx + 10, cy - 34, 7, theme.secondary);
    tft.fillCircle(cx, cy - 40, 20, theme.ok);
    tft.fillCircle(cx, cy - 50, 4, theme.primary);
  } else { // lvl >= 4 (Quantum Bloom)
    tft.fillRect(cx - 3, cy - 32, 6, 38, theme.ok);
    tft.fillCircle(cx - 22, cy - 10, 11, theme.ok);
    tft.fillCircle(cx + 22, cy - 10, 11, theme.ok);
    tft.fillCircle(cx - 20, cy - 24, 9, theme.primary);
    tft.fillCircle(cx + 20, cy - 24, 9, theme.primary);
    tft.fillCircle(cx - 12, cy - 36, 8, theme.secondary);
    tft.fillCircle(cx + 12, cy - 36, 8, theme.secondary);
    
    tft.fillCircle(cx - 15, cy - 42, 9, RGB565(255, 120, 200));
    tft.fillCircle(cx + 15, cy - 42, 9, RGB565(255, 120, 200));
    tft.fillCircle(cx, cy - 54, 9, RGB565(255, 120, 200));
    tft.fillCircle(cx, cy - 42, 21, theme.ok);

    iconCrown(cx, cy - 58, RGB565(255, 215, 0));
  }

  // 3. Мимика и эмоции
  int headY = (lvl == 1) ? (cy - 24) : (lvl == 2) ? (cy - 35) : (lvl == 3) ? (cy - 40) : (cy - 42);

  if (mood == MOOD_LOVE) {
    drawVectorHeart(cx - 7, headY - 1, 4, theme.accent);
    drawVectorHeart(cx + 7, headY - 1, 4, theme.accent);
    tft.fillCircle(cx - 12, headY + 5, 3, RGB565(255, 130, 180));
    tft.fillCircle(cx + 12, headY + 5, 3, RGB565(255, 130, 180));
    tft.drawFastHLine(cx - 4, headY + 7, 9, theme.bg);
    drawVectorHeart(cx - 14, headY - 24, 4, RGB565(255, 75, 130));
    drawVectorHeart(cx + 14, headY - 28, 5, RGB565(255, 75, 130));
  } else if (mood == MOOD_HAPPY) {
    tft.fillCircle(cx - 7, headY - 1, 3, theme.bg);
    tft.fillCircle(cx + 7, headY - 1, 3, theme.bg);
    tft.fillCircle(cx - 6, headY - 2, 1, theme.txtMain);
    tft.fillCircle(cx + 8, headY - 2, 1, theme.txtMain);
    tft.fillCircle(cx - 12, headY + 5, 2, theme.warn);
    tft.fillCircle(cx + 12, headY + 5, 2, theme.warn);
    tft.drawFastHLine(cx - 4, headY + 7, 9, theme.bg);
  } else if (mood == MOOD_CHILL) {
    tft.drawFastHLine(cx - 8, headY - 1, 5, theme.bg);
    tft.drawFastHLine(cx + 4, headY - 1, 5, theme.bg);
    tft.drawFastHLine(cx - 3, headY + 6, 6, theme.bg);
  } else { // MOOD_THIRSTY
    tft.fillCircle(cx - 7, headY + 1, 3, theme.bg);
    tft.fillCircle(cx + 7, headY + 1, 3, theme.bg);
    drawVectorDrop(cx + 14, headY - 7, 3, theme.secondary);
    tft.drawFastHLine(cx - 4, headY + 9, 9, theme.bg);
  }

  if (pumpState) {
    drawVectorDrop(cx - 10, cy + 4, 3, theme.secondary);
    drawVectorDrop(cx + 10, cy + 2, 3, theme.secondary);
  }
}

// ==========================================
// ЭКРАН ТАМАГОЧИ (CYBER SPROUT)
// ==========================================
void drawSproutScreen() {
  tft.fillScreen(theme.bg);
  drawMinimalHeader(tr("CYBER SPROUT", "КИБЕР-РОСТОК"), false);

  bool aiActive = isAiConnected();
  int health = aiActive ? (int)aiPlantHealth : calculatePlantHealthScore();
  if (aiActive) {
    sproutLevel = constrain(aiGrowthStage, 1, 4);
    int xpMax = sproutLevel * 100;
    sproutXP = constrain((int)((aiGrowthProgress * (float)xpMax) / 100.0f), 0, xpMax);
  }

  if (millis() < sproutLoveUntil) {
    currentSproutMood = MOOD_LOVE;
  } else if (health >= 85) {
    currentSproutMood = MOOD_HAPPY;
  } else if (health >= 65) {
    currentSproutMood = MOOD_CHILL;
  } else {
    currentSproutMood = MOOD_THIRSTY;
  }

  // 1. Верхний информационный статус-бар питомца со шкалой EXP
  drawGlowCard(8, 40, 304, 28, theme.primary, theme.surface, theme.border);
  tft.setTextSize(1);
  tft.setTextColor(theme.primary);
  tft.setCursor(14, 50);
  tft.print("LVL " + String(sproutLevel));

  // Шкала опыта
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(54, 50);
  tft.print("XP:");
  int xpMax = sproutLevel * 100;
  tft.fillRoundRect(74, 49, 74, 8, 3, theme.surfaceHi);
  int fillXP = constrain((74 * sproutXP) / xpMax, 0, 74);
  tft.fillRoundRect(74, 49, fillXP, 8, 3, theme.accent);

  tft.setCursor(154, 50);
  tft.print(String(sproutXP) + "/" + String(xpMax));

  const char* moodNamesEN[] = { "THRIVING", "CONTENT", "NEEDS CARE", "IN LOVE!" };
  const char* moodNamesRU[] = { "ОТЛИЧНО", "ДОВОЛЕН", "ЖАЖДА", "СЧАСТЛИВ!" };
  uint16_t moodColors[] = { theme.ok, theme.primary, theme.alert, theme.accent };
  const char* moodText = (currentLang == LANG_RU) ? moodNamesRU[currentSproutMood] : moodNamesEN[currentSproutMood];
  drawPill(216, 44, 88, 20, moodText, moodColors[currentSproutMood], theme.surfaceHi);

  // 2. Область персонажа (Слева, x: 8..144, y: 72..200)
  drawCard(8, 72, 138, 128, theme.surface, theme.border);
  drawSproutCharacter(77, 152, sproutLevel, currentSproutMood);

  // 3. Диалоговый бабл с мудростью агронома (Справа, x: 152..312, y: 72..122)
  drawGlowCard(152, 72, 160, 50, theme.secondary, theme.surface, theme.border);
  tft.fillTriangle(152, 86, 152, 96, 146, 91, theme.border);
  tft.fillTriangle(153, 87, 153, 95, 147, 91, theme.surface);

  if (aiActive && aiDiagnosis.length() > 0) {
    String bubbleMsg = aiCropName + ": " + aiDiagnosis + " (" + String((int)aiConfidence) + "%). " + tr("Growth: ", "Рост: ") + String((int)aiGrowthProgress) + "%";
    drawWrappedText(bubbleMsg.c_str(), 158, 78, 148, 10, 3, theme.txtMain);
  } else {
    const char* quote = (currentLang == LANG_RU) ? agroWisdomQuotesRU[currentWisdomIndex] : agroWisdomQuotesEN[currentWisdomIndex];
    drawWrappedText(quote, 158, 78, 148, 10, 3, theme.txtMain);
  }

  // 4. Интерактивные кнопки действий питомца (Справа, y: 126..202)
  // Кнопка 1: [ ❤️ ПОГЛАДИТЬ / PET SPROUT ]
  drawCard(152, 126, 160, 23, theme.surfaceHi, theme.borderHi);
  drawVectorHeart(164, 137, 4, RGB565(255, 75, 130));
  tft.setTextSize(1);
  tft.setTextColor(theme.txtMain);
  tft.setCursor(174, 133);
  tft.print(tr("PET SPROUT (+5XP)", "ПОГЛАДИТЬ (+5XP)"));

  // Кнопка 2: [ 💧 ПОЛИТЬ / WATER & FEED ]
  drawCard(152, 152, 160, 23, theme.surfaceHi, theme.borderHi);
  drawVectorDrop(164, 163, 4, theme.secondary);
  tft.setTextColor(theme.txtMain);
  tft.setCursor(174, 159);
  tft.print(tr("WATER & FEED (+10XP)", "ПОЛИТЬ (+10XP)"));

  // Кнопка 3: [ 🔄 СБРОС УРОВНЯ / RESET SPROUT LVL ]
  drawCard(152, 178, 160, 24, theme.surfaceHi, theme.borderHi);
  iconDot(164, 190, 3, theme.warn);
  tft.setTextSize(1);
  tft.setTextColor(theme.warn);
  tft.setCursor(174, 186);
  tft.print(tr("RESET SPROUT LVL", "СБРОС УРОВНЯ"));

  drawBottomTabs(2);
}

// ==========================================
// АРКАДНАЯ МИНИ-ИГРА (HYDRO CATCHER)
// ==========================================
void startMiniGame() {
  currentScreen = SCR_GAME;
  gameScore = 0;
  gameActive = true;
  gameOver = false;
  gameStartTime = millis();
  basketX = 160;
  prevBasketX = 160;
  lastGameTick = millis();

  for (int i = 0; i < MAX_DROPS; i++) {
    gameDrops[i].x = random(30, SCREEN_W - 30);
    gameDrops[i].y = -15 - i * 45;
    gameDrops[i].vy = 2.4f + (float)i * 0.4f;
    gameDrops[i].prevX = -1;
    gameDrops[i].prevY = -1;
    gameDrops[i].type = (i % 3 == 0) ? 1 : (i % 3 == 1) ? 0 : 2;
    gameDrops[i].active = true;
  }

  tft.fillScreen(theme.bg);

  // Шапка игры
  tft.fillRect(0, 0, SCREEN_W, 30, theme.surface);
  tft.drawFastHLine(0, 30, SCREEN_W, theme.border);

  tft.setTextSize(1);
  tft.setTextColor(theme.primary);
  tft.setCursor(12, 11);
  tft.print("TIME: 30s");

  tft.setTextColor(theme.txtMain);
  tft.setCursor(110, 11);
  tft.print("SCORE: 0");

  tft.setTextColor(theme.txtMuted);
  tft.setCursor(195, 11);
  tft.print("BEST: " + String(highScoreGame));

  // Кнопка выхода [X]
  drawCard(286, 5, 26, 20, theme.surfaceHi, theme.borderHi);
  tft.setTextColor(theme.alert);
  tft.setCursor(295, 10);
  tft.print("X");
}

void exitMiniGame() {
  gameActive = false;
  gameOver = false;
  currentScreen = SCR_SPROUT;
  drawSproutScreen();
}

void endMiniGame() {
  gameActive = false;
  gameOver = true;

  bool isNewRecord = false;
  if (gameScore > highScoreGame) {
    highScoreGame = gameScore;
    isNewRecord = true;
  }
  int earnedXP = max(10, gameScore / 4);
  addSproutXP(earnedXP);

  drawGlowCard(30, 36, 260, 172, theme.primary, theme.surface, theme.border);

  tft.setTextSize(2);
  tft.setTextColor(theme.primary);
  tft.setCursor(75, 48);
  tft.print("GAME OVER!");

  tft.setTextSize(1);
  tft.setTextColor(theme.txtMain);
  tft.setCursor(65, 80);
  tft.print("YOUR SCORE:  " + String(gameScore) + " PTS");

  tft.setTextColor(isNewRecord ? theme.ok : theme.txtMuted);
  tft.setCursor(65, 96);
  if (isNewRecord) {
    tft.print("*** NEW HIGH SCORE! ***");
  } else {
    tft.print("BEST RECORD: " + String(highScoreGame) + " PTS");
  }

  tft.setTextColor(theme.accent);
  tft.setCursor(65, 118);
  tft.print("SPROUT EARNED: +" + String(earnedXP) + " EXP!");

  // Кнопка [PLAY AGAIN]
  drawCard(46, 146, 108, 30, theme.surfaceHi, theme.borderHi);
  tft.setTextColor(theme.primary);
  tft.setCursor(56, 157);
  tft.print("PLAY AGAIN");

  // Кнопка [EXIT TO SPROUT]
  drawCard(166, 146, 108, 30, theme.surfaceHi, theme.borderHi);
  tft.setTextColor(theme.txtMain);
  tft.setCursor(182, 157);
  tft.print("BACK HOME");
}

void updateMiniGame() {
  if (!gameActive) return;
  unsigned long now = millis();
  if (now - lastGameTick < 30) return;
  lastGameTick = now;

  int elapsedSec = (now - gameStartTime) / 1000;
  int secLeft = 30 - elapsedSec;
  if (secLeft <= 0) {
    endMiniGame();
    return;
  }

  static int lastSecLeft = -1;
  static int lastScoreShown = -1;
  if (secLeft != lastSecLeft) {
    tft.fillRect(48, 11, 30, 10, theme.surface);
    tft.setTextSize(1);
    tft.setTextColor(secLeft <= 5 ? theme.alert : theme.primary);
    tft.setCursor(48, 11);
    tft.print(String(secLeft) + "s");
    lastSecLeft = secLeft;
  }
  if (gameScore != lastScoreShown) {
    tft.fillRect(152, 11, 36, 10, theme.surface);
    tft.setTextSize(1);
    tft.setTextColor(theme.txtMain);
    tft.setCursor(152, 11);
    tft.print(String(gameScore));
    lastScoreShown = gameScore;
  }

  // Обновление положения корзины
  if (basketX != prevBasketX) {
    tft.fillRect(prevBasketX - 24, 204, 48, 16, theme.bg);
    tft.fillRoundRect(basketX - 22, 204, 44, 14, 4, theme.surfaceHi);
    tft.drawRoundRect(basketX - 22, 204, 44, 14, 4, theme.primary);
    tft.fillRoundRect(basketX - 16, 211, 32, 4, 2, theme.primary);
    prevBasketX = basketX;
  }

  // Обновление падающих элементов
  for (int i = 0; i < MAX_DROPS; i++) {
    if (gameDrops[i].prevX >= 0) {
      tft.fillCircle(gameDrops[i].prevX, gameDrops[i].prevY, 6, theme.bg);
    }

    gameDrops[i].y += gameDrops[i].vy;

    // Попадание в корзину
    if (gameDrops[i].y >= 196 && gameDrops[i].y <= 212 && abs((int)gameDrops[i].x - basketX) < 26) {
      if (gameDrops[i].type == 0) {
        gameScore += 10;
      } else if (gameDrops[i].type == 1) {
        gameScore += 25;
      } else {
        gameScore = max(0, gameScore - 15);
      }
      gameDrops[i].x = random(30, SCREEN_W - 30);
      gameDrops[i].y = -15 - random(0, 20);
      gameDrops[i].type = (random(0, 10) < 4) ? 0 : (random(0, 10) < 7) ? 1 : 2;
    } else if (gameDrops[i].y > 224) {
      gameDrops[i].x = random(30, SCREEN_W - 30);
      gameDrops[i].y = -15 - random(0, 20);
      gameDrops[i].type = (random(0, 10) < 4) ? 0 : (random(0, 10) < 7) ? 1 : 2;
    }

    int curDX = (int)gameDrops[i].x;
    int curDY = (int)gameDrops[i].y;
    if (curDY >= 35 && curDY <= 220) {
      if (gameDrops[i].type == 0) {
        drawVectorDrop(curDX, curDY, 4, theme.secondary);
      } else if (gameDrops[i].type == 1) {
        tft.fillCircle(curDX, curDY, 5, theme.ok);
        tft.fillCircle(curDX, curDY, 2, theme.txtMain);
      } else {
        tft.fillCircle(curDX, curDY, 4, theme.alert);
        tft.drawFastHLine(curDX - 3, curDY, 7, theme.bg);
        tft.drawFastVLine(curDX, curDY - 3, 7, theme.bg);
      }
      gameDrops[i].prevX = curDX;
      gameDrops[i].prevY = curDY;
    } else {
      gameDrops[i].prevX = -1;
      gameDrops[i].prevY = -1;
    }
  }
}

// ==========================================
// РЕЖИМ НОЧНОГО ФИТО-СВЕТИЛЬНИКА (AMBIENT LAMP)
// ==========================================
void drawLampScreen() {
  uint32_t duty = (lampDuties[lampBrightnessIdx] * 255) / 100;
#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
  ledcWrite(TFT_BL, duty);
#else
  ledcWrite(PWM_BL_CH, duty);
#endif

  tft.fillScreen(lampColors[lampColorIdx]);

  // Плавающая панель управления
  drawCard(12, 194, 296, 36, RGB565(12, 14, 18), RGB565(40, 50, 65));

  drawPill(18, 200, 120, 24, lampNames[lampColorIdx], RGB565(245, 250, 255), RGB565(26, 34, 46));

  String bStr = "BRT: " + String(lampDuties[lampBrightnessIdx]) + "%";
  drawPill(146, 200, 80, 24, bStr.c_str(), RGB565(245, 250, 255), RGB565(26, 34, 46));

  drawPill(234, 200, 68, 24, "EXIT", RGB565(255, 80, 80), RGB565(36, 22, 26));
}

void exitLampMode() {
  setDisplayBrightness(brightnessLevel);
  currentScreen = SCR_SETTINGS;
  drawSettingsScreen();
}

// ==========================================
// ЭКРАН СМАРТ-АГРОНОМА (DIAGNOSTICS & ADVISOR)
// ==========================================
void drawAdvisorScreen() {
  tft.fillScreen(theme.bg);
  drawMinimalHeader(tr("AGRO ADVISOR", "АГРО-СОВЕТНИК"), true);

  bool aiActive = isAiConnected();
  int health = aiActive ? (int)aiPlantHealth : calculatePlantHealthScore();
  uint16_t hColor = (health >= 85) ? theme.ok : (health >= 65 ? theme.warn : theme.alert);

  drawGlowCard(8, 42, 304, 46, hColor, theme.surface, theme.border);
  tft.setTextSize(3);
  tft.setTextColor(hColor);
  tft.setCursor(18, 52);
  tft.print(String(health) + "%");

  tft.setTextSize(1);
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(100, 50);
  tft.print(aiActive ? tr("AI DIAG: ", "AI ДИАГНОЗ: ") : tr("STATUS: ", "СТАТУС: "));
  tft.setTextColor(hColor);
  if (aiActive) {
    tft.print(aiDiagnosis + " [" + aiStageName + "]");
  } else {
    tft.print(health >= 85 ? tr("OPTIMAL ECOSYSTEM", "ОПТИМАЛЬНАЯ СИСТЕМА") : (health >= 65 ? tr("SLIGHT IMBALANCE", "ЛЕГКИЙ ДИСБАЛАНС") : tr("ATTENTION REQUIRED", "ТРЕБУЕТ ВНИМАНИЯ")));
  }

  tft.setTextColor(theme.txtDim);
  tft.setCursor(100, 68);
  if (aiActive) {
    tft.print(tr("Growth: ", "Рост: ") + String((int)aiGrowthProgress) + "% | " + tr("Biomass: ", "Биомасса: ") + String(aiBiomass, 1) + "%");
  } else {
    tft.print("VPD " + String(current_VPD, 2) + " kPa | " + tr("Transpiration in zone", "Транспирация в норме"));
  }

  int cardY = 94;

  // Совет 1: pH & Питание
  drawCard(8, cardY, 304, 38, theme.surface, theme.border);
  iconDot(18, cardY + 14, 3, (current_pH >= PH_MIN_OK && current_pH <= PH_MAX_OK) ? theme.ok : theme.warn);
  tft.setTextSize(1);
  tft.setTextColor(theme.txtMain);
  tft.setCursor(28, cardY + 8);
  tft.print(tr("NUTRIENT SOLUTION ABSORPTION", "УСВОЕНИЕ ПИТАТЕЛЬНЫХ ВЕЩЕСТВ"));
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(28, cardY + 22);
  if (current_pH > PH_MAX_OK) {
    tft.print(tr("pH is high. Risk of iron lockout!", "pH высокий. Блокировка железа!"));
  } else if (current_pH < PH_MIN_OK) {
    tft.print(tr("pH is low. Risk of calcium deficiency!", "pH низкий. Дефицит кальция!"));
  } else {
    tft.print(tr("pH optimal for N-P-K micronutrients.", "pH идеален для усвоения N-P-K."));
  }

  // Совет 2: Корневая зона и температура воды
  cardY += 44;
  drawCard(8, cardY, 304, 38, theme.surface, theme.border);
  bool wOk = (current_waterTemp >= WT_MIN_OK && current_waterTemp <= WT_MAX_OK);
  iconDot(18, cardY + 14, 3, wOk ? theme.ok : theme.warn);
  tft.setTextSize(1);
  tft.setTextColor(theme.txtMain);
  tft.setCursor(28, cardY + 8);
  tft.print(tr("ROOT OXYGENATION & HEALTH", "КИСЛОРОД И ЗДОРОВЬЕ КОРНЕЙ"));
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(28, cardY + 22);
  if (current_waterTemp > WT_MAX_OK) {
    tft.print(tr("High water temp. Lower dissolved O2!", "Теплая вода. Падает растворенный O2!"));
  } else {
    tft.print(tr("Water temp optimal for root respiration.", "Температура воды идеальна для корней."));
  }

  // Совет 3: VPD и климат
  cardY += 44;
  drawCard(8, cardY, 304, 46, theme.surface, theme.border);
  bool vpdOk = (current_VPD >= VPD_MIN_OK && current_VPD <= VPD_MAX_OK);
  iconDot(18, cardY + 14, 3, vpdOk ? theme.ok : theme.warn);
  tft.setTextSize(1);
  tft.setTextColor(theme.txtMain);
  tft.setCursor(28, cardY + 8);
  tft.print(tr("TRANSPIRATION (VPD)", "ТРАНСПИРАЦИЯ (VPD)"));
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(28, cardY + 22);
  if (current_VPD < VPD_MIN_OK) {
    tft.print(tr("Low VPD: stomata close, fungal risk!", "Низкий VPD: риск развития грибков!"));
  } else if (current_VPD > VPD_MAX_OK) {
    tft.print(tr("High VPD: water stress on foliage!", "Высокий VPD: водный стресс листьев!"));
  } else {
    tft.print(tr("Optimal VPD: peak photosynthate flow.", "Оптимальный VPD: активный рост."));
  }
}

// ==========================================
// ЭКРАН EDGE AI & ДИАГНОСТИКИ РАСТЕНИЯ (SCR_VISION)
// ==========================================
void drawVisionScreen() {
  tft.fillScreen(theme.bg);
  drawMinimalHeader(tr("AI PLANT VISION", "AI ЗРЕНИЕ"), false);

  bool aiActive = isAiConnected();
  int health = aiActive ? (int)aiPlantHealth : calculatePlantHealthScore();
  uint16_t hColor = (health >= 85) ? theme.ok : (health >= 65 ? theme.warn : theme.alert);

  // Статусная полоса источника данных (y: 38..48)
  tft.setTextSize(1);
  if (aiActive) {
    iconDot(14, 43, 3, theme.ok);
    tft.setTextColor(theme.ok);
    tft.setCursor(24, 40);
    tft.print(tr("RPi 4B EDGE AI ONLINE", "RPi 4B AI НА СВЯЗИ"));

    tft.setTextColor(theme.txtDim);
    tft.setCursor(174, 40);
    tft.print("CPU " + String((int)aiRpiTemp) + "C | " + String((int)aiInferenceMs) + "ms");
  } else {
    iconDot(14, 43, 3, theme.warn);
    tft.setTextColor(theme.warn);
    tft.setCursor(24, 40);
    tft.print(tr("STANDBY: AUTONOMOUS SENSORS", "АВТОНОМНЫЕ ДАТЧИКИ"));

    tft.setTextColor(theme.txtDim);
    tft.setCursor(200, 40);
    tft.print(tr("USB/WiFi IDLE", "СВЯЗЬ ЖДЕТ"));
  }

  // Ряд 1: Здоровье AI (слева) и Культура / Диагноз (справа) (y: 50..102)
  drawGlowCard(8, 50, 118, 52, hColor, theme.surface, theme.border);
  tft.setTextSize(1);
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(16, 56);
  tft.print(tr("AI HEALTH", "AI ЗДОРОВЬЕ"));

  tft.setTextSize(3);
  tft.setTextColor(hColor);
  tft.setCursor(16, 69);
  tft.print(String(health) + "%");

  drawGlowCard(132, 50, 180, 52, theme.secondary, theme.surface, theme.border);
  tft.setTextSize(1);
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(140, 56);
  tft.print(tr("CROP: ", "КУЛЬТУРА: "));
  tft.setTextColor(theme.txtMain);
  tft.print(aiActive ? aiCropName : tr("Agro Culture", "Агро-культура"));

  tft.setCursor(140, 69);
  tft.setTextColor(hColor);
  String diagStr = aiActive ? (aiDiagnosis + " (" + String((int)aiConfidence) + "%)") : tr("Sensor Normal", "Норма датчиков");
  if (utf8_char_count(diagStr.c_str()) > 22) diagStr = diagStr.substring(0, 22);
  tft.print(diagStr);

  tft.setCursor(140, 84);
  tft.setTextColor(theme.txtDim);
  tft.print(tr("Sev: ", "Уровень: ") + (aiActive ? aiSeverity : tr("None", "Нет")) + " | " + (aiSeverity == "None" ? tr("Clean", "Чисто") : tr("Alert", "Внимание")));

  // Ряд 2: Прогресс роста растения и биомасса (y: 106..148)
  drawGlowCard(8, 106, 304, 42, theme.primary, theme.surface, theme.border);
  tft.setTextSize(1);
  tft.setTextColor(theme.primary);
  tft.setCursor(16, 111);
  tft.print(tr("GROWTH: ", "РОСТ: ") + String((int)aiGrowthProgress) + "%");

  tft.setTextColor(theme.txtMuted);
  tft.setCursor(136, 111);
  String stageStr = (aiGrowthStage == 1 ? tr("SPROUT", "ПРОРОСТОК") : (aiGrowthStage == 2 ? tr("VEG", "ВЕГЕТАЦИЯ") : (aiGrowthStage == 3 ? tr("BLOOM", "ЦВЕТЕНИЕ") : tr("MATURE", "УРОЖАЙ"))));
  tft.print(tr("STAGE ", "СТАДИЯ ") + String(aiGrowthStage) + "/4: " + stageStr);

  // Многосегментный прогресс-бар развития растения
  int barW = 288;
  int barH = 7;
  tft.fillRoundRect(16, 122, barW, barH, 3, theme.surfaceHi);
  int fillW = constrain((int)((aiGrowthProgress * (float)barW) / 100.0f), 4, barW);
  tft.fillRoundRect(16, 122, fillW, barH, 3, theme.primary);

  // Разделители стадий (Проросток / Вегетация / Цветение / Урожай)
  for (int st = 1; st <= 3; st++) {
    int dx = 16 + (st * barW) / 4;
    tft.drawFastVLine(dx, 122, barH, theme.borderHi);
  }

  tft.setCursor(16, 134);
  tft.setTextColor(theme.txtDim);
  tft.print(tr("Biomass: ", "Биомасса: ") + String(aiBiomass, 1) + "% | " + tr("Chl: ", "Хлор: ") + String(aiChlorosis, 1) + "% | " + tr("Nec: ", "Некр: ") + String(aiNecrosis, 1) + "%");

  // Ряд 3: Агрономические рекомендации от Edge AI (y: 152..198)
  drawCard(8, 152, 304, 46, theme.surface, theme.border);
  iconDot(18, 161, 3, theme.accent);
  tft.setTextSize(1);
  tft.setTextColor(theme.accent);
  tft.setCursor(26, 156);
  tft.print(tr("AI RECOMMENDATION:", "AI РЕКОМЕНДАЦИЯ:"));

  drawWrappedText(aiAdvice.c_str(), 26, 168, 280, 9, 3, theme.txtMain);

  // Навигационная панель табов (Таб 1: AI VISION / ИИ ЗРЕНИЕ)
  drawBottomTabs(1);
}

// ==========================================
// ПАРСИНГ JSON И ДВУСТОРОННЯЯ СВЯЗЬ С RASPBERRY PI
// ==========================================
String getJsonString(const String& json, const String& key) {
  String pattern = "\"" + key + "\":";
  int idx = json.indexOf(pattern);
  if (idx == -1) return "";
  idx += pattern.length();
  while (idx < (int)json.length() && json[idx] == ' ') idx++;
  if (idx < (int)json.length() && json[idx] == '\"') {
    idx++;
    int endIdx = json.indexOf("\"", idx);
    if (endIdx == -1) return "";
    return json.substring(idx, endIdx);
  }
  int endIdx = idx;
  while (endIdx < (int)json.length() && json[endIdx] != ',' && json[endIdx] != '}' && json[endIdx] != ' ') {
    endIdx++;
  }
  return json.substring(idx, endIdx);
}

float getJsonFloat(const String& json, const String& key, float defaultVal) {
  String pattern = "\"" + key + "\":";
  int idx = json.indexOf(pattern);
  if (idx == -1) return defaultVal;
  idx += pattern.length();
  while (idx < (int)json.length() && (json[idx] == ' ' || json[idx] == '\"')) idx++;
  int endIdx = idx;
  while (endIdx < (int)json.length() && (isDigit(json[endIdx]) || json[endIdx] == '.' || json[endIdx] == '-')) endIdx++;
  if (endIdx > idx) {
    return json.substring(idx, endIdx).toFloat();
  }
  return defaultVal;
}

int getJsonInt(const String& json, const String& key, int defaultVal) {
  return (int)getJsonFloat(json, key, (float)defaultVal);
}

void applyAiSyncData(const String& jsonStr) {
  aiPlantHealth   = constrain(getJsonFloat(jsonStr, "ai_health", aiPlantHealth), 0.0f, 100.0f);
  aiGrowthProgress = constrain(getJsonFloat(jsonStr, "growth", aiGrowthProgress), 0.0f, 100.0f);
  aiGrowthStage   = constrain(getJsonInt(jsonStr, "stage", aiGrowthStage), 1, 4);

  String sName = getJsonString(jsonStr, "stage_name");
  if (sName.length() > 0) aiStageName = sName;

  aiBiomass       = getJsonFloat(jsonStr, "biomass", aiBiomass);
  aiChlorosis     = getJsonFloat(jsonStr, "chl", aiChlorosis);
  aiNecrosis      = getJsonFloat(jsonStr, "nec", aiNecrosis);

  String crop = getJsonString(jsonStr, "crop");
  if (crop.length() > 0) aiCropName = crop;

  String diag = getJsonString(jsonStr, "diag");
  if (diag.length() > 0) aiDiagnosis = diag;

  aiConfidence    = getJsonFloat(jsonStr, "conf", aiConfidence);

  String sev = getJsonString(jsonStr, "sev");
  if (sev.length() > 0) aiSeverity = sev;

  String adv = getJsonString(jsonStr, "advice");
  if (adv.length() > 0) aiAdvice = adv;

  aiRpiTemp       = getJsonFloat(jsonStr, "rpi_temp", aiRpiTemp);
  aiInferenceMs   = getJsonFloat(jsonStr, "infer_ms", aiInferenceMs);

  aiLastSyncMillis = millis();
  aiConnected     = true;
  aiHasEverSynced = true;

  // Автоматическая синхронизация тамагочи с реальным развитием растения
  sproutLevel = constrain(aiGrowthStage, 1, 4);
  int xpMax = sproutLevel * 100;
  sproutXP = constrain((int)((aiGrowthProgress * (float)xpMax) / 100.0f), 0, xpMax);

  if (currentScreen == SCR_VISION && !isSleeping) {
    drawVisionScreen();
  } else if (currentScreen == SCR_HOME && !isSleeping) {
    refreshHomeValues();
  } else if (currentScreen == SCR_SPROUT && !isSleeping) {
    drawSproutScreen();
  }
}

String serialRxBuffer = "";
unsigned long lastSerialTelemetry = 0;

void sendSerialTelemetry(const char* type) {
  String msg = "{\"type\":\"" + String(type) + "\",";
  msg += "\"ph\":" + String(current_pH, 2) + ",";
  msg += "\"tds\":" + String(current_TDS) + ",";
  msg += "\"water_temp\":" + String(current_waterTemp, 2) + ",";
  msg += "\"air_temp\":" + String(current_airTemp, 2) + ",";
  msg += "\"humidity\":" + String(current_humidity, 2) + ",";
  msg += "\"vpd\":" + String(current_VPD, 2) + ",";
  msg += "\"pump\":" + String(pumpState ? "true" : "false") + ",";
  msg += "\"health_calc\":" + String(calculatePlantHealthScore()) + ",";
  msg += "\"uptime\":" + String(millis() / 1000) + "}\n";
  Serial.print(msg);
}

void processSerialCommunication() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      serialRxBuffer.trim();
      if (serialRxBuffer.length() > 0) {
        if (serialRxBuffer.startsWith("{") && serialRxBuffer.endsWith("}")) {
          String type = getJsonString(serialRxBuffer, "type");
          if (type == "ai_sync") {
            applyAiSyncData(serialRxBuffer);
            sendSerialTelemetry("ack");
          } else if (type == "cmd") {
            String action = getJsonString(serialRxBuffer, "action");
            if (action == "pump_on") { pumpState = true; lastPumpState = true; }
            else if (action == "pump_off") { pumpState = false; lastPumpState = false; }
            else if (action == "toggle_pump") { pumpState = !pumpState; lastPumpState = pumpState; }
            else if (action == "pet") { sproutLoveUntil = millis() + 6000; currentWisdomIndex = (currentWisdomIndex + 1) % WISDOM_COUNT; }
            else if (action == "reset_sprout") { resetSproutProgress(); }
            digitalWrite(PUMP_PIN, pumpState ? HIGH : LOW);
            if (currentScreen == SCR_HOME && !isSleeping) refreshHomeValues();
            if (currentScreen == SCR_SPROUT && !isSleeping) drawSproutScreen();
            sendSerialTelemetry("ack");
          }
        }
        serialRxBuffer = "";
      }
    } else {
      if (serialRxBuffer.length() < 1024) {
        serialRxBuffer += c;
      } else {
        serialRxBuffer = "";
      }
    }
  }

  // Периодическая трансляция телеметрии в Serial (каждые 2 секунды)
  unsigned long now = millis();
  if (now - lastSerialTelemetry >= 2000UL) {
    lastSerialTelemetry = now;
    sendSerialTelemetry("telemetry");
  }
}

// ==========================================
// ЛОКАЛЬНЫЙ WEB DASHBOARD (MOBILE & PC)
// ==========================================
void setupWebDashboard() {
  webServer.on("/", HTTP_GET, []() {
    String html = "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
                  "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                  "<title>AgroBox Cyber-Station</title><style>"
                  ":root{--bg:#0a0e14;--card:#121924;--border:#243248;--primary:#00f5b9;--text:#f8fafc;--muted:#8c9eb6}"
                  "body{margin:0;padding:20px;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,sans-serif}"
                  ".box{max-width:520px;margin:0 auto;background:var(--card);padding:24px;border-radius:20px;border:1px solid var(--border);box-shadow:0 12px 36px rgba(0,0,0,0.6)}"
                  ".hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}"
                  "h1{margin:0;font-size:24px;color:var(--primary);letter-spacing:0.04em}"
                  ".badge{background:#1c2636;color:#00f5b9;padding:6px 12px;border-radius:12px;font-size:12px;font-weight:700;border:1px solid #2e405a}"
                  ".sub{color:var(--muted);font-size:13px;margin-bottom:20px}"
                  ".grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px}"
                  ".card{background:#182232;padding:14px;border-radius:14px;border:1px solid var(--border);position:relative;overflow:hidden}"
                  ".card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--primary),transparent)}"
                  ".card h3{margin:0 0 6px;font-size:11px;color:var(--muted);letter-spacing:0.06em}"
                  ".val{font-size:26px;font-weight:700;color:var(--text)}"
                  ".unit{font-size:12px;color:var(--muted);font-weight:400;margin-left:4px}"
                  ".sprout-box{background:#16202e;border:1px solid #2a3a52;padding:14px;border-radius:14px;margin-bottom:20px;display:flex;align-items:center;gap:12px}"
                  ".sprout-avatar{font-size:32px}"
                  ".sprout-msg{font-size:13px;color:#c0d0e4;font-style:italic}"
                  ".actions{display:flex;gap:10px}"
                  ".btn{flex:1;padding:14px;font-size:15px;font-weight:600;text-align:center;border-radius:12px;cursor:pointer;border:none;background:var(--primary);color:#0a0e14;transition:0.2s}"
                  ".btn.off{background:#28354a;color:var(--text)}"
                  ".btn.sec{background:#8c64ff;color:#fff}"
                  "</style></head><body><div class='box'>"
                  "<div class='hdr'><h1>AGROBOX CYBER</h1><div style='display:flex;align-items:center;gap:6px;'><div class='badge' id='healthBadge'>--% PRIME</div><button class='btn sec' style='padding:4px 8px;font-size:11px' onclick='setLang(\"en\")'>EN</button><button class='btn sec' style='padding:4px 8px;font-size:11px' onclick='setLang(\"ru\")'>RU</button></div></div>"
                  "<div class='sub'>Automated Hydroponics & CyberSprout Hub</div>"
                  "<div class='sprout-box'><div class='sprout-avatar'>🌱</div><div><strong id='sproutMood'>CyberSprout</strong><div class='sprout-msg' id='wisdom'>\"Growing with high-tech ions!\"</div></div></div>"
                  "<div class='grid'>"
                  "<div class='card'><h3 id='phTitle'>pH LEVEL</h3><div class='val' id='ph'>--</div></div>"
                  "<div class='card'><h3 id='tdsTitle'>TDS NUTRIENTS</h3><div class='val' id='tds'>--<span class='unit'>ppm</span></div></div>"
                  "<div class='card'><h3 id='wtTitle'>WATER TEMP</h3><div class='val' id='wt'>--<span class='unit'>°C</span></div></div>"
                  "<div class='card'><h3 id='atTitle'>AIR TEMP</h3><div class='val' id='at'>--<span class='unit'>°C</span></div></div>"
                  "<div class='card'><h3 id='humTitle'>HUMIDITY</h3><div class='val' id='hum'>--<span class='unit'>%</span></div></div>"
                  "<div class='card'><h3 id='vpdTitle'>VPD DEFICIT</h3><div class='val' id='vpd'>--<span class='unit'>kPa</span></div></div>"
                  "</div>"
                  "<div class='card' style='margin-bottom:18px;border:1px solid #00f5b9'>"
                  "<div style='display:flex;justify-content:space-between;align-items:center'>"
                  "<h3 style='color:#00f5b9;margin:0;font-size:12px;letter-spacing:0.05em'>EDGE AI PLANT VISION (RPI 4B)</h3>"
                  "<span class='badge' id='aiBadge'>STANDBY</span>"
                  "</div>"
                  "<div style='margin-top:10px;font-size:15px;font-weight:700'>"
                  "AI Health: <span id='aiHealth' style='color:#00f5b9'>--</span>% | Growth: <span id='aiGrowth'>--</span>% (<span id='aiStage'>--</span>)"
                  "</div>"
                  "<div style='margin-top:6px;font-size:13px;color:#f8fafc'>"
                  "Diagnosis: <span id='aiDiag' style='color:#82e0ff'>--</span>"
                  "</div>"
                  "<div style='margin-top:4px;font-size:12px;color:#c0d0e4;font-style:italic' id='aiAdvice'>--</div>"
                  "</div>"
                  "<div class='actions'>"
                  "<button class='btn' id='pumpBtn' onclick='togglePump()'>Pump</button>"
                  "<button class='btn sec' onclick='petSprout()'>❤️ Pet Sprout</button>"
                  "</div>"
                  "</div><script>"
                  "async function poll(){"
                  "try{const r=await fetch('/api/data');const d=await r.json();"
                  "document.getElementById('ph').innerText=d.ph.toFixed(1);"
                  "document.getElementById('tds').innerHTML=d.tds+'<span class=\"unit\">ppm</span>';"
                  "document.getElementById('wt').innerHTML=d.waterTemp.toFixed(1)+'<span class=\"unit\">°C</span>';"
                  "document.getElementById('at').innerHTML=d.airTemp.toFixed(1)+'<span class=\"unit\">°C</span>';"
                  "document.getElementById('hum').innerHTML=Math.round(d.humidity)+'<span class=\"unit\">%</span>';"
                  "document.getElementById('vpd').innerHTML=d.vpd.toFixed(2)+'<span class=\"unit\">kPa</span>';"
                  "if(d.lang=='ru'){"
                  "document.getElementById('phTitle').innerText='УРОВЕНЬ pH';"
                  "document.getElementById('tdsTitle').innerText='ПИТАНИЕ (TDS)';"
                  "document.getElementById('wtTitle').innerText='ТЕМП. ВОДЫ';"
                  "document.getElementById('atTitle').innerText='ТЕМП. ВОЗДУХА';"
                  "document.getElementById('humTitle').innerText='ВЛАЖНОСТЬ';"
                  "document.getElementById('vpdTitle').innerText='VPD (ДЕФИЦИТ)';"
                  "document.getElementById('sproutMood').innerText='Кибер-Росток';"
                  "document.getElementById('healthBadge').innerText=d.health+'% НОРМА';"
                  "}else{"
                  "document.getElementById('phTitle').innerText='pH LEVEL';"
                  "document.getElementById('tdsTitle').innerText='TDS NUTRIENTS';"
                  "document.getElementById('wtTitle').innerText='WATER TEMP';"
                  "document.getElementById('atTitle').innerText='AIR TEMP';"
                  "document.getElementById('humTitle').innerText='HUMIDITY';"
                  "document.getElementById('vpdTitle').innerText='VPD DEFICIT';"
                  "document.getElementById('sproutMood').innerText='CyberSprout';"
                  "document.getElementById('healthBadge').innerText=d.health+'% PRIME';"
                  "}"
                  "document.getElementById('aiHealth').innerText=d.ai_health?d.ai_health.toFixed(1):d.health;"
                  "document.getElementById('aiGrowth').innerText=d.ai_growth?Math.round(d.ai_growth):'--';"
                  "document.getElementById('aiStage').innerText=d.ai_stage||'Veg';"
                  "document.getElementById('aiDiag').innerText=d.ai_diag?(d.ai_crop+': '+d.ai_diag+' ('+Math.round(d.ai_conf)+'%)'):'Healthy';"
                  "document.getElementById('aiAdvice').innerText=d.ai_advice||'Optimal conditions.';"
                  "const b=document.getElementById('aiBadge');b.innerText=d.ai_online?'ONLINE':'STANDBY';"
                  "b.style.borderColor=d.ai_online?'#00f5b9':'#ffb142';b.style.color=d.ai_online?'#00f5b9':'#ffb142';"
                  "const btn=document.getElementById('pumpBtn');"
                  "if(d.pump){btn.innerText='Pump: ACTIVE';btn.className='btn'}"
                  "else{btn.innerText='Pump: STANDBY';btn.className='btn off'}"
                  "}catch(e){}}"
                  "async function setLang(l){await fetch('/api/setLanguage?lang='+l);poll();}"
                  "async function togglePump(){await fetch('/api/togglePump');poll();}"
                  "async function petSprout(){await fetch('/api/pet');alert('You petted CyberSprout! ❤️');}"
                  "setInterval(poll,1500);poll();"
                  "</script></body></html>";
    webServer.send(200, "text/html", html);
  });

  webServer.on("/api/data", HTTP_GET, []() {
    bool aiActive = isAiConnected();
    int health = aiActive ? (int)aiPlantHealth : calculatePlantHealthScore();

    String json = "{";
    json += "\"ph\":" + String(current_pH, 2) + ",";
    json += "\"tds\":" + String(current_TDS) + ",";
    json += "\"waterTemp\":" + String(current_waterTemp, 2) + ",";
    json += "\"airTemp\":" + String(current_airTemp, 2) + ",";
    json += "\"humidity\":" + String(current_humidity, 2) + ",";
    json += "\"vpd\":" + String(current_VPD, 2) + ",";
    json += "\"health\":" + String(health) + ",";
    json += "\"pump\":" + String(pumpState ? "true" : "false") + ",";
    json += "\"ai_health\":" + String(aiPlantHealth, 1) + ",";
    json += "\"ai_growth\":" + String(aiGrowthProgress, 1) + ",";
    json += "\"ai_stage\":\"" + aiStageName + "\",";
    json += "\"ai_crop\":\"" + aiCropName + "\",";
    json += "\"ai_diag\":\"" + aiDiagnosis + "\",";
    json += "\"ai_conf\":" + String(aiConfidence, 1) + ",";
    json += "\"ai_advice\":\"" + aiAdvice + "\",";
    json += "\"ai_online\":" + String(aiActive ? "true" : "false") + ",";
    json += "\"lang\":\"" + String(currentLang == LANG_RU ? "ru" : "en") + "\"";
    json += "}";
    webServer.send(200, "application/json", json);
  });

  webServer.on("/api/setLanguage", HTTP_ANY, []() {
    if (webServer.hasArg("lang")) {
      String l = webServer.arg("lang");
      if (l == "ru" || l == "RU") {
        currentLang = LANG_RU;
      } else {
        currentLang = LANG_EN;
      }
      saveLanguagePreference();
      if (!isSleeping) {
        if (currentScreen == SCR_SETTINGS) drawSettingsScreen();
        else if (currentScreen == SCR_HOME) drawHomeScreen();
        else if (currentScreen == SCR_SPROUT) drawSproutScreen();
        else if (currentScreen == SCR_VISION) drawVisionScreen();
        else if (currentScreen == SCR_DETAIL) drawDetailScreen();
        else if (currentScreen == SCR_ADVISOR) drawAdvisorScreen();
      }
      webServer.send(200, "application/json", "{\"status\":\"ok\",\"lang\":\"" + String(currentLang == LANG_RU ? "ru" : "en") + "\"}");
    } else {
      webServer.send(400, "application/json", "{\"status\":\"error\",\"msg\":\"missing lang\"}");
    }
  });

  // REST API: прием телеметрии Edge AI от Raspberry Pi по Wi-Fi
  webServer.on("/api/ai/update", HTTP_POST, []() {
    if (webServer.hasArg("plain")) {
      String body = webServer.arg("plain");
      applyAiSyncData(body);
      webServer.send(200, "application/json", "{\"status\":\"ok\"}");
    } else {
      webServer.send(400, "application/json", "{\"status\":\"error\",\"msg\":\"missing body\"}");
    }
  });

  // REST API: передача данных сенсоров инкубатора для RPi
  webServer.on("/api/rpi/telemetry", HTTP_GET, []() {
    String json = "{";
    json += "\"ph\":" + String(current_pH, 2) + ",";
    json += "\"tds\":" + String(current_TDS) + ",";
    json += "\"water_temp\":" + String(current_waterTemp, 2) + ",";
    json += "\"air_temp\":" + String(current_airTemp, 2) + ",";
    json += "\"humidity\":" + String(current_humidity, 2) + ",";
    json += "\"vpd\":" + String(current_VPD, 2) + ",";
    json += "\"pump\":" + String(pumpState ? "true" : "false") + ",";
    json += "\"health_calc\":" + String(calculatePlantHealthScore()) + ",";
    json += "\"ai_health\":" + String(aiPlantHealth, 1) + ",";
    json += "\"growth\":" + String(aiGrowthProgress, 1) + ",";
    json += "\"uptime\":" + String(millis() / 1000);
    json += "}";
    webServer.send(200, "application/json", json);
  });

  // REST API: удаленное управление (помпа, свет) от Raspberry Pi
  webServer.on("/api/cmd", HTTP_POST, []() {
    if (webServer.hasArg("plain")) {
      String body = webServer.arg("plain");
      String action = getJsonString(body, "action");
      if (action == "pump_on") { pumpState = true; lastPumpState = true; }
      else if (action == "pump_off") { pumpState = false; lastPumpState = false; }
      else if (action == "toggle_pump") { pumpState = !pumpState; lastPumpState = pumpState; }
      else if (action == "reset_sprout") { resetSproutProgress(); }
      if (currentScreen == SCR_HOME) refreshHomeValues();
      else if (currentScreen == SCR_SPROUT) drawSproutScreen();
      webServer.send(200, "application/json", "{\"status\":\"ok\",\"pump\":" + String(pumpState ? "true" : "false") + "}");
    } else {
      webServer.send(400, "application/json", "{\"status\":\"error\"}");
    }
  });

  webServer.on("/api/togglePump", HTTP_GET, []() {
    pumpState = !pumpState;
    lastPumpState = pumpState;
    if (currentScreen == SCR_HOME) refreshHomeValues();
    webServer.send(200, "application/json", "{\"status\":\"ok\",\"pump\":" + String(pumpState ? "true" : "false") + "}");
  });

  webServer.on("/api/pet", HTTP_GET, []() {
    sproutLoveUntil = millis() + 6000;
    currentWisdomIndex = (currentWisdomIndex + 1) % WISDOM_COUNT;
    if (currentScreen == SCR_SPROUT) drawSproutScreen();
    webServer.send(200, "application/json", "{\"status\":\"ok\",\"pet\":\"happy\"}");
  });

  webServer.begin();
  webServerStarted = true;
}

// ==========================================
// ДИНАМИЧЕСКИЙ 3D-СКРИНСЕЙВЕР (WARP STARFIELD)
// ==========================================
void initStarfield() {
  for (int i = 0; i < STAR_COUNT; i++) {
    stars[i].x = (float)(random(-160, 160));
    stars[i].y = (float)(random(-120, 120));
    stars[i].z = (float)(random(20, 260));
    stars[i].prevSX = -1;
    stars[i].prevSY = -1;
  }
}

void enterSleepMode() {
  isSleeping = true;
  initStarfield();
  tft.fillScreen(theme.bg);

  // Снижаем яркость дисплея до 12%
  uint32_t duty = (12 * 255) / 100;
#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
  ledcWrite(TFT_BL, duty);
#else
  ledcWrite(PWM_BL_CH, duty);
#endif
}

void updateStarfield() {
  static unsigned long lastStarTick = 0;
  if (millis() - lastStarTick < 35) return;
  lastStarTick = millis();

  for (int i = 0; i < STAR_COUNT; i++) {
    // Стираем предыдущую точку
    if (stars[i].prevSX >= 0 && stars[i].prevSY >= 0) {
      tft.drawPixel(stars[i].prevSX, stars[i].prevSY, theme.bg);
    }

    stars[i].z -= 4.5f;
    if (stars[i].z <= 10.0f) {
      stars[i].x = (float)(random(-160, 160));
      stars[i].y = (float)(random(-120, 120));
      stars[i].z = 240.0f;
    }

    int sx = 160 + (int)(stars[i].x * 120.0f / stars[i].z);
    int sy = 120 + (int)(stars[i].y * 120.0f / stars[i].z);

    if (sx < 0 || sx >= SCREEN_W || sy < 0 || sy >= SCREEN_H) {
      stars[i].z = 10.0f;
      stars[i].prevSX = -1;
      stars[i].prevSY = -1;
    } else {
      uint16_t col = (stars[i].z < 70) ? theme.txtMain : (stars[i].z < 150 ? theme.primary : theme.txtDim);
      tft.drawPixel(sx, sy, col);
      stars[i].prevSX = sx;
      stars[i].prevSY = sy;
    }
  }

  // Обновление мягкого ночного HUD поверх звездопада
  static unsigned long lastSleepHud = 0;
  if (millis() - lastSleepHud >= 1000) {
    lastSleepHud = millis();
    int health = calculatePlantHealthScore();

    // Верхняя ночная плашка
    drawGlowCard(40, 24, 240, 48, theme.primary, theme.surface, theme.border);
    tft.setTextSize(3);
    tft.setTextColor(theme.txtMain);
    tft.setCursor(55, 34);
    tft.print(String(health) + "% OK");

    tft.setTextSize(1);
    tft.setTextColor(theme.primary);
    tft.setCursor(185, 36);
    tft.print("AGROBOX");
    tft.setTextColor(theme.txtMuted);
    tft.setCursor(185, 48);
    tft.print("VPD " + String(current_VPD, 2));

    // Нижняя подпись
    tft.fillRect(70, 196, 180, 16, theme.bg);
    tft.setTextColor(theme.txtDim);
    tft.setCursor(85, 200);
    tft.print("- TOUCH TO ENGAGE -");
  }
}

void wakeFromSleepMode() {
  isSleeping = false;
  setDisplayBrightness(brightnessLevel);
  if (currentScreen == SCR_HOME) drawHomeScreen();
  else if (currentScreen == SCR_SPROUT) drawSproutScreen();
  else if (currentScreen == SCR_DETAIL) drawDetailScreen();
  else if (currentScreen == SCR_SETTINGS) drawSettingsScreen();
  else if (currentScreen == SCR_ADVISOR) drawAdvisorScreen();
  else if (currentScreen == SCR_VISION) drawVisionScreen();
}

// ==========================================
// ГЛАВНЫЙ ЭКРАН (HOME / MONITOR)
// ==========================================
void drawHomeScreen() {
  tft.fillScreen(theme.bg);
  drawMinimalHeader("AGROBOX", false);

  bool phOk  = (current_pH >= PH_MIN_OK && current_pH <= PH_MAX_OK);
  bool tdsOk = (current_TDS >= TDS_MIN_OK && current_TDS <= TDS_MAX_OK);

  // Ряд 1: pH (слева) со спарклайном
  drawGlowCard(8, 42, 148, 52, phOk ? theme.primary : theme.warn, theme.surface, theme.border);
  iconDot(18, 52, 3, phOk ? theme.ok : theme.warn);
  tft.setTextSize(1);
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(26, 49);
  tft.print(tr("pH LEVEL", "УРОВЕНЬ pH"));

  tft.setTextSize(2);
  tft.setTextColor(theme.txtMain);
  tft.setCursor(18, 68);
  tft.print(String(current_pH, 1));

  // Спарклайн тренда pH
  drawSparkline(74, 66, 46, 18, metricHistory[METRIC_PH], 12, theme.primary);
  drawPill(124, 66, 26, 18, phOk ? "OK" : "!", phOk ? theme.ok : theme.warn, theme.surfaceHi);

  // Ряд 1: TDS (справа) со спарклайном
  drawGlowCard(164, 42, 148, 52, tdsOk ? theme.primary : theme.warn, theme.surface, theme.border);
  iconDot(174, 52, 3, tdsOk ? theme.ok : theme.warn);
  tft.setTextSize(1);
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(182, 49);
  tft.print(tr("NUTRIENTS (TDS)", "ПИТАНИЕ (TDS)"));

  tft.setTextSize(2);
  tft.setTextColor(theme.txtMain);
  tft.setCursor(174, 68);
  tft.print(String(current_TDS));

  drawSparkline(226, 66, 46, 18, metricHistory[METRIC_TDS], 12, theme.secondary);
  drawPill(276, 66, 30, 18, "ppm", theme.txtMuted, theme.surfaceHi);

  // Ряд 2: 4-модульный блок климата со светящейся окантовкой
  drawGlowCard(8, 100, 304, 46, theme.secondary, theme.surface, theme.border);

  // 1. Water Temp
  bool wtOk = (current_waterTemp >= WT_MIN_OK && current_waterTemp <= WT_MAX_OK);
  tft.setTextSize(1);
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(16, 107);
  tft.print(tr("WATER", "ВОДА"));
  tft.setTextSize(2);
  tft.setTextColor(wtOk ? theme.txtMain : theme.warn);
  tft.setCursor(16, 122);
  tft.print(String(current_waterTemp, 1));

  tft.drawFastVLine(82, 107, 32, theme.border);

  // 2. Air Temp
  bool atOk = (current_airTemp >= AT_MIN_OK && current_airTemp <= AT_MAX_OK);
  tft.setTextSize(1);
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(92, 107);
  tft.print(tr("AIR", "ВОЗДУХ"));
  tft.setTextSize(2);
  tft.setTextColor(atOk ? theme.txtMain : theme.warn);
  tft.setCursor(92, 122);
  tft.print(String(current_airTemp, 1));

  tft.drawFastVLine(158, 107, 32, theme.border);

  // 3. Humidity
  bool humOk = (current_humidity >= HUM_MIN_OK && current_humidity <= HUM_MAX_OK);
  tft.setTextSize(1);
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(168, 107);
  tft.print(tr("HUMID", "ВЛАЖН"));
  tft.setTextSize(2);
  tft.setTextColor(humOk ? theme.txtMain : theme.warn);
  tft.setCursor(168, 122);
  tft.print(String((int)current_humidity) + "%");

  tft.drawFastVLine(234, 107, 32, theme.border);

  // 4. VPD
  bool vpdOk = (current_VPD >= VPD_MIN_OK && current_VPD <= VPD_MAX_OK);
  tft.setTextSize(1);
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(244, 107);
  tft.print(tr("VPD kPa", "VPD кПа"));
  tft.setTextSize(2);
  tft.setTextColor(vpdOk ? theme.primary : theme.warn);
  tft.setCursor(244, 122);
  tft.print(String(current_VPD, 2));

  // Ряд 3: Помпа с интерактивным тумблером и каналом потока
  drawGlowCard(8, 152, 304, 46, pumpState ? theme.ok : 0, theme.surface, theme.border);
  iconPower(26, 175, 7, pumpState ? theme.ok : theme.txtDim);
  tft.setTextSize(2);
  tft.setTextColor(theme.txtMain);
  tft.setCursor(44, 160);
  tft.print(tr("PUMP", "ПОМПА"));

  tft.setTextSize(1);
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(44, 180);
  if (pumpAutoMode) {
    tft.print(pumpState ? tr("Feed", "Полив") : tr("Rest", "Пауза"));
  } else {
    tft.print(pumpState ? tr("Run", "Работа") : tr("Stop", "Стоп"));
  }

  // Канал циркуляции жидкости
  if (!pumpState) {
    tft.fillRoundRect(104, 180, 132, 7, 3, theme.surfaceHi);
    tft.drawFastHLine(104, 183, 132, theme.border);
  }

  drawModernToggle(248, 161, 48, 26, pumpState);

  drawBottomTabs(0);

  // Синхронизация кэша
  lastDrawn_pH = current_pH;
  lastDrawn_TDS = current_TDS;
  lastDrawn_waterTemp = current_waterTemp;
  lastDrawn_airTemp = current_airTemp;
  lastDrawn_humidity = current_humidity;
  lastDrawn_VPD = current_VPD;
  lastDrawn_Health = isAiConnected() ? (int)aiPlantHealth : calculatePlantHealthScore();
  lastPumpState = pumpState;
  lastWifiConnected = (WiFi.status() == WL_CONNECTED);
}

void refreshHomeValues() {
  if (currentScreen != SCR_HOME) return;

  bool connected = (WiFi.status() == WL_CONNECTED);
  bool aiActive = isAiConnected();
  int health = aiActive ? (int)aiPlantHealth : calculatePlantHealthScore();
  if (connected != lastWifiConnected || health != lastDrawn_Health) {
    drawMinimalHeader("AGROBOX", false);
    lastWifiConnected = connected;
    lastDrawn_Health = health;
  }

  // pH
  if (current_pH != lastDrawn_pH) {
    bool phOk = (current_pH >= PH_MIN_OK && current_pH <= PH_MAX_OK);
    tft.fillRect(18, 68, 50, 18, theme.surface);
    tft.setTextSize(2);
    tft.setTextColor(theme.txtMain);
    tft.setCursor(18, 68);
    tft.print(String(current_pH, 1));
    tft.fillRect(74, 66, 46, 18, theme.surface);
    drawSparkline(74, 66, 46, 18, metricHistory[METRIC_PH], 12, theme.primary);
    drawPill(124, 66, 26, 18, phOk ? "OK" : "!", phOk ? theme.ok : theme.warn, theme.surfaceHi);
    lastDrawn_pH = current_pH;
  }

  // TDS
  if (current_TDS != lastDrawn_TDS) {
    tft.fillRect(174, 68, 50, 18, theme.surface);
    tft.setTextSize(2);
    tft.setTextColor(theme.txtMain);
    tft.setCursor(174, 68);
    tft.print(String(current_TDS));
    tft.fillRect(226, 66, 46, 18, theme.surface);
    drawSparkline(226, 66, 46, 18, metricHistory[METRIC_TDS], 12, theme.secondary);
    lastDrawn_TDS = current_TDS;
  }

  // Water Temp
  if (current_waterTemp != lastDrawn_waterTemp) {
    tft.fillRect(16, 122, 60, 18, theme.surface);
    tft.setTextSize(2);
    tft.setTextColor(theme.txtMain);
    tft.setCursor(16, 122);
    tft.print(String(current_waterTemp, 1));
    lastDrawn_waterTemp = current_waterTemp;
  }

  // Air Temp
  if (current_airTemp != lastDrawn_airTemp) {
    tft.fillRect(92, 122, 60, 18, theme.surface);
    tft.setTextSize(2);
    tft.setTextColor(theme.txtMain);
    tft.setCursor(92, 122);
    tft.print(String(current_airTemp, 1));
    lastDrawn_airTemp = current_airTemp;
  }

  // Humidity
  if (current_humidity != lastDrawn_humidity) {
    tft.fillRect(168, 122, 60, 18, theme.surface);
    tft.setTextSize(2);
    tft.setTextColor(theme.txtMain);
    tft.setCursor(168, 122);
    tft.print(String((int)current_humidity) + "%");
    lastDrawn_humidity = current_humidity;
  }

  // VPD
  if (current_VPD != lastDrawn_VPD) {
    tft.fillRect(244, 122, 60, 18, theme.surface);
    tft.setTextSize(2);
    tft.setTextColor(theme.primary);
    tft.setCursor(244, 122);
    tft.print(String(current_VPD, 2));
    lastDrawn_VPD = current_VPD;
  }

  // Помпа
  if (pumpState != lastPumpState) {
    drawModernToggle(248, 161, 48, 26, pumpState);
    iconPower(26, 175, 7, pumpState ? theme.ok : theme.txtDim);
    tft.fillRect(44, 180, 56, 12, theme.surface);
    tft.setTextSize(1);
    tft.setTextColor(theme.txtMuted);
    tft.setCursor(44, 180);
    if (pumpAutoMode) {
      tft.print(pumpState ? tr("Feed", "Полив") : tr("Rest", "Пауза"));
    } else {
      tft.print(pumpState ? tr("Run", "Работа") : tr("Stop", "Стоп"));
    }
    if (!pumpState) {
      tft.fillRoundRect(104, 180, 132, 7, 3, theme.surfaceHi);
      tft.drawFastHLine(104, 183, 132, theme.border);
    }
    lastPumpState = pumpState;
  }
}

// ==========================================
// КАРТОЧКА РЕГУЛИРОВКИ ЯРКОСТИ (SETTINGS)
// ==========================================
void drawBrightnessControlCard(int y) {
  drawCard(8, y, 304, 32, theme.surface, theme.border);
  tft.setTextSize(1);
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(16, y + 12);
  tft.print(tr("BRIGHTNESS:", "ЯРКОСТЬ:"));

  // Кнопка [-]
  drawCard(94, y + 6, 22, 20, theme.surfaceHi, theme.borderHi);
  tft.setTextSize(2);
  tft.setTextColor(theme.txtMain);
  tft.setCursor(101, y + 9);
  tft.print("-");

  // Прогресс-бар
  int barX = 122, barY = y + 12, barW = 88, barH = 8;
  tft.fillRoundRect(barX, barY, barW, barH, 4, theme.surfaceHi);
  int fillW = (barW * brightnessLevel) / 100;
  tft.fillRoundRect(barX, barY, fillW, barH, 4, theme.primary);

  // Кнопка [+]
  drawCard(216, y + 6, 22, 20, theme.surfaceHi, theme.borderHi);
  tft.setCursor(222, y + 9);
  tft.print("+");

  // Процент
  tft.setTextSize(1);
  tft.setTextColor(theme.txtMain);
  tft.setCursor(246, y + 12);
  tft.print(String(brightnessLevel) + "% ");
}

void drawSettingsScreen() {
  tft.fillScreen(theme.bg);
  drawMinimalHeader(tr("SETTINGS", "НАСТРОЙКИ"), false);

  // 1. Wi-Fi Card с IP (y = 40, h = 34)
  drawGlowCard(8, 40, 304, 34, theme.primary, theme.surface, theme.border);
  iconSignal(18, 49, (WiFi.status() == WL_CONNECTED) ? 4 : 1, theme.ok, theme.borderHi);
  tft.setTextSize(1);
  tft.setTextColor(theme.txtMain);
  tft.setCursor(44, 45);
  tft.print(tr("Wi-Fi & Web Station", "Wi-Fi и Веб-станция"));

  tft.setTextColor(theme.txtMuted);
  tft.setCursor(44, 58);
  if (WiFi.status() == WL_CONNECTED) {
    tft.print("http://" + WiFi.localIP().toString());
  } else if (savedSSID.length() > 0) {
    tft.print("Saved: " + savedSSID);
  } else {
    tft.print(tr("Tap to configure", "Нажмите для настройки"));
  }
  iconChevronRight(288, 50, 10, theme.txtDim);

  // 2. Регулировка Яркости (y = 76, h = 32)
  drawBrightnessControlCard(76);

  // 3. Выбор языка (Language: EN / RU) (y = 110, h = 28)
  drawCard(8, 110, 304, 28, theme.surface, theme.border);
  tft.setTextSize(1);
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(16, 120);
  tft.print(tr("LANGUAGE:", "ЯЗЫК:"));

  bool isEn = (currentLang == LANG_EN);
  bool isRu = (currentLang == LANG_RU);
  drawPill(108, 114, 90, 20, "ENGLISH", isEn ? theme.bg : theme.txtMuted, isEn ? theme.primary : theme.surfaceHi);
  drawPill(206, 114, 98, 20, "РУССКИЙ", isRu ? theme.bg : theme.txtMuted, isRu ? theme.primary : theme.surfaceHi);

  // 4. Выбор темы оформления (Themes: Cyber / Nordic / Solar) (y = 140, h = 28)
  drawCard(8, 140, 304, 28, theme.surface, theme.border);
  tft.setTextSize(1);
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(16, 150);
  tft.print(tr("THEME:", "ТЕМА:"));

  const char* themeNamesEN[3] = { "CYBER", "NORDIC", "SOLAR" };
  const char* themeNamesRU[3] = { "КИБЕР", "НОРДИК", "СОЛАР" };
  for (int i = 0; i < 3; i++) {
    int btnX = 72 + i * 78;
    bool active = ((int)currentTheme == i);
    const char* tName = (currentLang == LANG_RU) ? themeNamesRU[i] : themeNamesEN[i];
    drawPill(btnX, 144, 72, 20, tName, active ? theme.bg : theme.txtMuted, active ? theme.primary : theme.surfaceHi);
  }

  // 5. Опции: Demo Waves, Grow Lamp & Calibrate (y = 170, h = 28)
  drawCard(8, 170, 96, 28, theme.surface, theme.border);
  tft.setTextSize(1);
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(14, 179);
  tft.print(tr("DEMO", "ДЕМО"));
  drawPill(50, 174, 50, 20, demoWavesEnabled ? tr("ON", "ВКЛ") : tr("OFF", "ВЫКЛ"), demoWavesEnabled ? theme.ok : theme.txtDim, theme.surfaceHi);

  drawCard(112, 170, 96, 28, theme.surface, theme.border);
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(118, 179);
  tft.print(tr("LAMP", "ЛАМПА"));
  drawPill(152, 174, 52, 20, tr("LIGHT", "СВЕТ"), theme.primary, theme.surfaceHi);

  drawCard(216, 170, 96, 28, theme.surface, theme.border);
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(222, 179);
  tft.print(tr("TOUCH", "ТАЧ"));
  drawPill(258, 174, 50, 20, tr("ALIGN", "ТЕСТ"), theme.txtMain, theme.surfaceHi);

  drawBottomTabs(3);
}

// ==========================================
// ЭКРАН СКАНИРОВАНИЯ WI-FI С ПРОКРУТКОЙ
// ==========================================
void drawWifiListOnly() {
  tft.fillRect(0, HEADER_H, SCREEN_W, SCREEN_H - HEADER_H, theme.bg);

  int shown = min(wifiNetworkCount, 16);
  if (shown == 0) {
    tft.setTextSize(1);
    tft.setTextColor(theme.txtMuted);
    tft.setCursor(95, 120);
    tft.print("No Wi-Fi networks found");
    return;
  }

  int maxDisplay = min(4, shown - wifiScrollOffset);
  for (int i = 0; i < maxDisplay; i++) {
    int netIdx = wifiScrollOffset + i;
    int idx = wifiOrder[netIdx];
    int y = 42 + i * 44;

    drawCard(8, y, 290, 38, theme.surface, theme.border);

    int rssi = WiFi.RSSI(idx);
    int bars = (rssi > -60) ? 4 : (rssi > -70) ? 3 : (rssi > -80) ? 2 : 1;
    iconSignal(18, y + 13, bars, theme.primary, theme.borderHi);

    tft.setTextSize(2);
    tft.setTextColor(theme.txtMain);
    tft.setCursor(44, y + 11);
    String ssid = WiFi.SSID(idx);
    if (ssid.length() > 14) ssid = ssid.substring(0, 14) + "..";
    tft.print(ssid);

    iconChevronRight(276, y + 14, 10, theme.txtDim);
  }

  // Скроллбар
  if (shown > 4) {
    int barX = 306, barY = 42, barH = 176;
    tft.fillRoundRect(barX, barY, 4, barH, 2, theme.surface);
    int thumbH = max(20, (4 * barH) / shown);
    int maxScroll = shown - 4;
    int thumbY = barY + (wifiScrollOffset * (barH - thumbH)) / maxScroll;
    tft.fillRoundRect(barX, thumbY, 4, thumbH, 2, theme.primary);
  }
}

void drawWifiScanScreen() {
  tft.fillScreen(theme.bg);
  drawMinimalHeader("SCANNING...", true);

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(60);

  wifiNetworkCount = WiFi.scanNetworks();
  int shown = min(wifiNetworkCount, 16);

  for (int i = 0; i < shown; i++) wifiOrder[i] = i;
  for (int i = 1; i < shown; i++) {
    int key = wifiOrder[i];
    int keyRssi = WiFi.RSSI(key);
    int j = i - 1;
    while (j >= 0 && WiFi.RSSI(wifiOrder[j]) < keyRssi) {
      wifiOrder[j + 1] = wifiOrder[j];
      j--;
    }
    wifiOrder[j + 1] = key;
  }

  drawMinimalHeader("SELECT NETWORK", true);
  wifiScrollOffset = 0;
  drawWifiListOnly();
}

// ==========================================
// ЭКРАН КЛАВИАТУРЫ
// ==========================================
void drawInputHeader() {
  tft.fillRect(0, 0, SCREEN_W, 40, theme.surface);
  tft.drawFastHLine(0, 40, SCREEN_W, theme.border);
  iconBackArrow(10, 13, 14, theme.primary);

  tft.setTextSize(1);
  tft.setTextColor(theme.primary);
  tft.setCursor(34, 6);
  tft.print("SSID: " + selectedSSID);

  tft.setTextSize(2);
  tft.setTextColor(theme.txtMain);
  tft.setCursor(34, 18);
  tft.print(inputPassword + "_");
}

void drawKeyboardScreen() {
  tft.fillScreen(theme.bg);
  drawInputHeader();
  for (uint8_t i = 0; i < kbdKeyCount; i++) {
    KeyRect &k = kbdKeys[i];
    uint16_t color = colorForKey(k.label);
    tft.fillRoundRect(k.x, k.y, k.w, k.h, 4, color);
    tft.drawRoundRect(k.x, k.y, k.w, k.h, 4, theme.border);
    tft.setTextSize(2);
    tft.setTextColor((color == theme.ok || color == theme.alert) ? theme.bg : theme.txtMain);
    tft.setCursor(k.x + 4, k.y + 11);
    tft.print(k.label);
  }
}

void enterKeyboard(KbdMode mode) {
  kbdMode = mode;
  const char* (*kbd)[10] = (mode == KBD_LOWER) ? kbd_lower : (mode == KBD_UPPER) ? kbd_upper : kbd_num;
  buildKeyboardLayout(kbd);
  drawKeyboardScreen();
}

// ==========================================
// ПОДКЛЮЧЕНИЕ К WI-FI
// ==========================================
void connectToWiFi() {
  tft.fillScreen(theme.bg);
  drawMinimalHeader("CONNECTING", false);

  tft.setTextSize(1);
  tft.setTextColor(theme.txtMuted);
  tft.setCursor(40, 95);
  tft.print("JOINING NETWORK");

  tft.setTextSize(2);
  tft.setTextColor(theme.primary);
  tft.setCursor(40, 115);
  tft.print(selectedSSID);

  WiFi.begin(selectedSSID.c_str(), inputPassword.c_str());
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 16) {
    delay(400);
    tft.print(".");
    attempts++;
  }

  tft.fillScreen(theme.bg);
  if (WiFi.status() == WL_CONNECTED) {
    saveWiFiCredentials(selectedSSID, inputPassword);
    drawPill(70, 105, 180, 30, "CONNECTED SUCCESSFULLY", theme.bg, theme.ok);
  } else {
    drawPill(70, 105, 180, 30, "CONNECTION FAILED", theme.txtMain, theme.alert);
  }
  delay(1200);

  currentScreen = SCR_HOME;
  drawHomeScreen();
}

// ==========================================
// ОБРАБОТКА НАЖАТИЙ И ЖЕСТОВ
// ==========================================
void handleTouches() {
  if (!newGestureAvailable) return;

  GestureType gesture = currentGesture;
  int x = gestureStartX;
  int y = gestureStartY;
  newGestureAvailable = false;

  // Если экран спал — пробуждаем и предотвращаем ложные срабатывания кнопок
  if (isSleeping) {
    wakeFromSleepMode();
    lastTouchActivity = millis();
    return;
  }
  lastTouchActivity = millis();

  // Навигация свайпами (Влево / Вправо)
  if (gesture == GESTURE_SWIPE_LEFT) {
    if (currentScreen == SCR_HOME) {
      currentScreen = SCR_VISION;
      drawVisionScreen();
      return;
    } else if (currentScreen == SCR_VISION) {
      currentScreen = SCR_SPROUT;
      drawSproutScreen();
      return;
    } else if (currentScreen == SCR_SPROUT) {
      currentScreen = SCR_SETTINGS;
      drawSettingsScreen();
      return;
    } else if (currentScreen == SCR_DETAIL) {
      currentDetailMetric = (MetricType)((currentDetailMetric + 1) % 6);
      drawDetailScreen();
      return;
    }
  } else if (gesture == GESTURE_SWIPE_RIGHT) {
    if (currentScreen == SCR_SETTINGS) {
      currentScreen = SCR_SPROUT;
      drawSproutScreen();
      return;
    } else if (currentScreen == SCR_SPROUT) {
      currentScreen = SCR_VISION;
      drawVisionScreen();
      return;
    } else if (currentScreen == SCR_VISION || currentScreen == SCR_DETAIL || currentScreen == SCR_ADVISOR) {
      currentScreen = SCR_HOME;
      drawHomeScreen();
      return;
    } else if (currentScreen == SCR_WIFI_SCAN) {
      currentScreen = SCR_SETTINGS;
      drawSettingsScreen();
      return;
    } else if (currentScreen == SCR_KBD) {
      currentScreen = SCR_WIFI_SCAN;
      drawWifiScanScreen();
      return;
    }
  }

  // Обработка тапов по экранам
  if (currentScreen == SCR_HOME) {
    if (gesture == GESTURE_TAP) {
      // Клик по бейджу здоровья в шапке -> открываем экран Edge AI Vision
      if (x >= 118 && x <= 212 && y < HEADER_H) {
        currentScreen = SCR_VISION;
        drawVisionScreen();
        return;
      }
      // 1. Тап по карточке pH (x: 8..156, y: 42..94)
      if (x >= 8 && x <= 156 && y >= 42 && y <= 94) {
        currentDetailMetric = METRIC_PH;
        currentScreen = SCR_DETAIL;
        drawDetailScreen();
        return;
      }
      // 2. Тап по карточке TDS (x: 164..312, y: 42..94)
      else if (x >= 164 && x <= 312 && y >= 42 && y <= 94) {
        currentDetailMetric = METRIC_TDS;
        currentScreen = SCR_DETAIL;
        drawDetailScreen();
        return;
      }
      // 3. Тап по 4-колоночному блоку климата (y: 100..146)
      else if (y >= 100 && y <= 146) {
        if (x >= 8 && x < 82) { // Вода
          currentDetailMetric = METRIC_WATER_TEMP;
          currentScreen = SCR_DETAIL;
          drawDetailScreen();
          return;
        } else if (x >= 82 && x < 158) { // Воздух
          currentDetailMetric = METRIC_AIR_TEMP;
          currentScreen = SCR_DETAIL;
          drawDetailScreen();
          return;
        } else if (x >= 158 && x < 234) { // Влажность
          currentDetailMetric = METRIC_HUMIDITY;
          currentScreen = SCR_DETAIL;
          drawDetailScreen();
          return;
        } else if (x >= 234 && x <= 312) { // VPD
          currentDetailMetric = METRIC_VPD;
          currentScreen = SCR_DETAIL;
          drawDetailScreen();
          return;
        }
      }
      // 4. Клик по карточке помпы
      else if (x >= 8 && x <= 312 && y >= 152 && y <= 198) {
        pumpState = !pumpState;
        drawModernToggle(248, 161, 48, 26, pumpState);
        lastPumpState = pumpState;
        refreshHomeValues();
      }
      // 5. Нижний навбар (MONITOR, VISION, SPROUT, SETTINGS)
      else if (y > SCREEN_H - NAV_H) {
        if (x >= 80 && x < 160) {
          currentScreen = SCR_VISION;
          drawVisionScreen();
        } else if (x >= 160 && x < 240) {
          currentScreen = SCR_SPROUT;
          drawSproutScreen();
        } else if (x >= 240) {
          currentScreen = SCR_SETTINGS;
          drawSettingsScreen();
        }
      }
    }
  }
  else if (currentScreen == SCR_SPROUT) {
    if (gesture == GESTURE_TAP) {
      // Кнопка 1: [ PET SPROUT ] (y: 124..149)
      if (x >= 150 && x <= 312 && y >= 122 && y <= 149) {
        addSproutXP(5);
        sproutLoveUntil = millis() + 5000;
        currentWisdomIndex = (currentWisdomIndex + 1) % WISDOM_COUNT;
        drawSproutScreen();
        return;
      }
      // Кнопка 2: [ WATER & FEED ] (y: 150..175)
      else if (x >= 150 && x <= 312 && y >= 150 && y <= 175) {
        addSproutXP(10);
        pumpState = true;
        lastPumpState = pumpState;
        pumpManualOffAt = millis() + 5000;
        sproutLoveUntil = millis() + 5000;
        currentWisdomIndex = (currentWisdomIndex + 1) % WISDOM_COUNT;
        drawSproutScreen();
        return;
      }
      // Кнопка 3: [ 🔄 СБРОС УРОВНЯ / RESET SPROUT LVL ] (y: 176..204)
      else if (x >= 150 && x <= 312 && y >= 176 && y <= 204) {
        resetSproutProgress();
        drawSproutScreen();
        return;
      }
      // Тап по персонажу -> поглаживание
      else if (x >= 8 && x <= 146 && y >= 70 && y <= 200) {
        addSproutXP(5);
        sproutLoveUntil = millis() + 5000;
        currentWisdomIndex = (currentWisdomIndex + 1) % WISDOM_COUNT;
        drawSproutScreen();
        return;
      }
      // Тап по речевому баблу -> следующая мысль
      else if (x >= 150 && x <= 312 && y >= 70 && y <= 124) {
        currentWisdomIndex = (currentWisdomIndex + 1) % WISDOM_COUNT;
        drawSproutScreen();
        return;
      }
      // Навбар (4 таба)
      else if (y > SCREEN_H - NAV_H) {
        if (x < 80) {
          currentScreen = SCR_HOME;
          drawHomeScreen();
        } else if (x >= 80 && x < 160) {
          currentScreen = SCR_VISION;
          drawVisionScreen();
        } else if (x >= 240) {
          currentScreen = SCR_SETTINGS;
          drawSettingsScreen();
        }
      }
    }
  }
  else if (currentScreen == SCR_GAME) {
    if (gesture == GESTURE_TAP || gesture == GESTURE_SWIPE_LEFT || gesture == GESTURE_SWIPE_RIGHT) {
      if (gameActive) {
        // Кнопка выхода [X] в правом верхнем углу
        if (x >= 280 && y <= 35) {
          exitMiniGame();
          return;
        }
        basketX = constrain(x, 26, SCREEN_W - 26);
      } else if (gameOver) {
        // [PLAY AGAIN] (x: 46..154, y: 146..176)
        if (x >= 40 && x <= 156 && y >= 142 && y <= 180) {
          startMiniGame();
          return;
        }
        // [BACK HOME] (x: 166..274, y: 146..176)
        else if (x >= 162 && x <= 280 && y >= 142 && y <= 180) {
          exitMiniGame();
          return;
        }
      }
    }
  }
  else if (currentScreen == SCR_LAMP) {
    if (gesture == GESTURE_TAP) {
      // Кнопка цвета (x: 18..138, y: 196..230)
      if (x >= 18 && x <= 138 && y >= 194 && y <= 230) {
        lampColorIdx = (lampColorIdx + 1) % 4;
        drawLampScreen();
        return;
      }
      // Кнопка яркости (x: 146..226, y: 196..230)
      else if (x >= 144 && x <= 226 && y >= 194 && y <= 230) {
        lampBrightnessIdx = (lampBrightnessIdx + 1) % 4;
        drawLampScreen();
        return;
      }
      // Кнопка выхода [EXIT] или тап по верхней области
      else if ((x >= 230 && x <= 308 && y >= 194 && y <= 230) || (y < 190)) {
        exitLampMode();
        return;
      }
    }
  }
  else if (currentScreen == SCR_ADVISOR) {
    if (gesture == GESTURE_TAP) {
      if (x < 50 && y < HEADER_H) {
        currentScreen = SCR_HOME;
        drawHomeScreen();
        return;
      } else if (y >= 42 && y <= 90) {
        currentScreen = SCR_VISION;
        drawVisionScreen();
        return;
      }
    }
  }
  else if (currentScreen == SCR_VISION) {
    if (gesture == GESTURE_TAP) {
      // 1. Нижний навбар (4 таба)
      if (y > SCREEN_H - NAV_H) {
        if (x < 80) {
          currentScreen = SCR_HOME;
          drawHomeScreen();
          return;
        } else if (x >= 160 && x < 240) {
          currentScreen = SCR_SPROUT;
          drawSproutScreen();
          return;
        } else if (x >= 240) {
          currentScreen = SCR_SETTINGS;
          drawSettingsScreen();
          return;
        }
      }
      // 2. Кнопка назад в шапке
      else if (x < 50 && y < HEADER_H) {
        currentScreen = SCR_HOME;
        drawHomeScreen();
        return;
      }
      // 3. Тап по карточке рекомендаций (y: 152..198) -> запрос нового снимка у RPi
      else if (x >= 8 && x <= 312 && y >= 152 && y <= 198) {
        Serial.println("{\"type\":\"cmd\",\"action\":\"diagnose\"}");
        drawPill(180, 156, 120, 20, tr("REQUESTED...", "ЗАПРОШЕНО..."), theme.ok, theme.surfaceHi, theme.primary);
        return;
      }
    }
  }
  else if (currentScreen == SCR_DETAIL) {
    if (gesture == GESTURE_TAP) {
      // Кнопка назад в шапке
      if (x < 50 && y < HEADER_H) {
        currentScreen = SCR_HOME;
        drawHomeScreen();
        return;
      }
      // Переключение интервала (1H, 6H, 24H)
      if (y >= 90 && y <= 116) {
        if (x >= 170 && x < 216) {
          currentDetailTF = TF_1H;
          drawDetailScreen();
        } else if (x >= 216 && x < 262) {
          currentDetailTF = TF_6H;
          drawDetailScreen();
        } else if (x >= 262 && x <= 312) {
          currentDetailTF = TF_24H;
          drawDetailScreen();
        }
      }
    }
  }
  else if (currentScreen == SCR_SETTINGS) {
    if (gesture == GESTURE_TAP) {
      // Нижний навбар (4 таба)
      if (y > SCREEN_H - NAV_H) {
        if (x < 80) {
          currentScreen = SCR_HOME;
          drawHomeScreen();
        } else if (x >= 80 && x < 160) {
          currentScreen = SCR_VISION;
          drawVisionScreen();
        } else if (x >= 160 && x < 240) {
          currentScreen = SCR_SPROUT;
          drawSproutScreen();
        }
      }
      // Wi-Fi Setup Card (y: 40..74)
      else if (x >= 8 && x <= 312 && y >= 40 && y < 76) {
        currentScreen = SCR_WIFI_SCAN;
        drawWifiScanScreen();
      }
      // Регулировка яркости (y: 76..108)
      else if (y >= 76 && y < 108) {
        if (x >= 14 && x <= 118) {
          if (brightnessLevel > 10) {
            setDisplayBrightness(brightnessLevel - 10);
            drawBrightnessControlCard(76);
          }
        } else if (x >= 214 && x <= 248) {
          if (brightnessLevel < 100) {
            setDisplayBrightness(brightnessLevel + 10);
            drawBrightnessControlCard(76);
          }
        } else if (x >= 118 && x <= 214) {
          int mapped = map(x, 118, 214, 10, 100);
          setDisplayBrightness(constrain(mapped, 10, 100));
          drawBrightnessControlCard(76);
        }
      }
      // Выбор языка (EN / RU) (y: 108..138)
      else if (y >= 108 && y < 138) {
        if (x >= 100 && x < 204) {
          currentLang = LANG_EN;
          saveLanguagePreference();
          drawSettingsScreen();
        } else if (x >= 204 && x <= 312) {
          currentLang = LANG_RU;
          saveLanguagePreference();
          drawSettingsScreen();
        }
      }
      // Выбор темы оформления (CYBER, NORDIC, SOLAR) (y: 138..168)
      else if (y >= 138 && y < 168) {
        if (x >= 64 && x < 146) {
          applyTheme(THEME_CYBER_EMERALD);
          saveThemePreference();
          drawSettingsScreen();
        } else if (x >= 146 && x < 228) {
          applyTheme(THEME_NORDIC_ICE);
          saveThemePreference();
          drawSettingsScreen();
        } else if (x >= 228 && x <= 310) {
          applyTheme(THEME_SOLAR_AMBER);
          saveThemePreference();
          drawSettingsScreen();
        }
      }
      // Опции внизу настроек: Demo Waves, Grow Lamp, Touch Calibrate (y: 168..202)
      else if (y >= 168 && y <= 202) {
        if (x >= 8 && x < 108) {
          demoWavesEnabled = !demoWavesEnabled;
          drawCard(8, 170, 96, 28, theme.surface, theme.border);
          tft.setTextSize(1);
          tft.setTextColor(theme.txtMuted);
          tft.setCursor(14, 179);
          tft.print(tr("DEMO", "ДЕМО"));
          drawPill(50, 174, 50, 20, demoWavesEnabled ? tr("ON", "ВКЛ") : tr("OFF", "ВЫКЛ"), demoWavesEnabled ? theme.ok : theme.txtDim, theme.surfaceHi);
        } else if (x >= 108 && x < 214) {
          currentScreen = SCR_LAMP;
          drawLampScreen();
        } else if (x >= 214 && x <= 312) {
          runCalibration();
          currentScreen = SCR_SETTINGS;
          drawSettingsScreen();
        }
      }
    }
  }
  else if (currentScreen == SCR_WIFI_SCAN) {
    int shown = min(wifiNetworkCount, 16);

    if (gesture == GESTURE_SWIPE_UP) {
      if (wifiScrollOffset + 4 < shown) {
        wifiScrollOffset++;
        drawWifiListOnly();
      }
    } else if (gesture == GESTURE_SWIPE_DOWN) {
      if (wifiScrollOffset > 0) {
        wifiScrollOffset--;
        drawWifiListOnly();
      }
    } else if (gesture == GESTURE_TAP) {
      if (x < 40 && y < HEADER_H) {
        currentScreen = SCR_SETTINGS;
        drawSettingsScreen();
        return;
      }
      int maxDisplay = min(4, shown - wifiScrollOffset);
      for (int i = 0; i < maxDisplay; i++) {
        int itemY = 42 + i * 44;
        if (x >= 8 && x <= 298 && y >= itemY && y <= itemY + 38) {
          int netIdx = wifiScrollOffset + i;
          selectedSSID = WiFi.SSID(wifiOrder[netIdx]);
          inputPassword = "";
          currentScreen = SCR_KBD;
          enterKeyboard(KBD_LOWER);
          return;
        }
      }
    }
  }
  else if (currentScreen == SCR_KBD) {
    if (gesture == GESTURE_TAP) {
      if (x < 32 && y < 40) {
        currentScreen = SCR_WIFI_SCAN;
        drawWifiScanScreen();
        return;
      }
      for (uint8_t i = 0; i < kbdKeyCount; i++) {
        KeyRect &k = kbdKeys[i];
        if (x < k.x || x > k.x + k.w || y < k.y || y > k.y + k.h) continue;

        String key = k.label;
        if (key == "<-" && inputPassword.length() > 0) {
          inputPassword.remove(inputPassword.length() - 1);
        } else if (key == "CLR") {
          inputPassword = "";
        } else if (key == "OK") {
          connectToWiFi();
          return;
        } else if (key == "123") {
          enterKeyboard(KBD_NUM); return;
        } else if (key == "abc") {
          enterKeyboard(KBD_LOWER); return;
        } else if (key == "a/A") {
          enterKeyboard(kbdMode == KBD_LOWER ? KBD_UPPER : KBD_LOWER); return;
        } else if (key == "SPACE") {
          if (inputPassword.length() < 24) inputPassword += " ";
        } else if (inputPassword.length() < 24) {
          inputPassword += key;
        }
        drawInputHeader();
        break;
      }
    }
  }
}

// ==========================================
// ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ (SETUP)
// ==========================================
void setup() {
  Serial.begin(115200);

  // Инициализация реле помпы
  pinMode(PUMP_PIN, OUTPUT);
  digitalWrite(PUMP_PIN, LOW);

  // Инициализация палитры и подсветки
  applyTheme(THEME_CYBER_EMERALD);
  initBacklight();
  loadThemePreference();
  loadDisplaySettings();
  loadLanguagePreference();

  // Инициализация SPI и ILI9341
  tftSPI.begin(TFT_SCK, TFT_MISO, TFT_MOSI, TFT_CS);
  tft.begin();
  tft.setRotation(1);
  tft.setTextWrap(false);

  // Инициализация сенсорной панели
  touchSPI.begin(TOUCH_SCK, TOUCH_MISO, TOUCH_MOSI, TOUCH_CS);
  ts.begin(touchSPI);
  ts.setRotation(1);

  if (loadCalibration()) {
    isCalibrated = true;
    tft.fillScreen(theme.bg);
  } else {
    runCalibration();
  }

  // Фоновое автоподключение Wi-Fi
  WiFi.mode(WIFI_STA);
  if (loadWiFiCredentials(savedSSID, savedPass)) {
    WiFi.begin(savedSSID.c_str(), savedPass.c_str());
  }

  loadSproutProgress();
  current_VPD = calculateVPD(current_airTemp, current_humidity);
  initMetricHistory();
  drawHomeScreen();
  lastTouchActivity = millis();

  xTaskCreatePinnedToCore(TaskCore1, "TouchLogic", 10000, NULL, 1, &LogicTask, 1);
}

// ==========================================
// ОСНОВНОЙ ЦИКЛ (LOOP)
// ==========================================
unsigned long lastSensorPoll = 0;

void loop() {
  processSerialCommunication();
  if (!isCalibrated) return;
  handleTouches();

  unsigned long now = millis();

  // Запуск локального Web-сервера после подключения к Wi-Fi
  if (WiFi.status() == WL_CONNECTED) {
    if (!webServerStarted) setupWebDashboard();
    webServer.handleClient();
  }

  // Проверка активности для 3D-скринсейвера (60 секунд без касаний)
  if (!isSleeping && (now - lastTouchActivity >= 60000UL)) {
    enterSleepMode();
  }

  if (isSleeping) {
    updateStarfield();
  } else {
    // Обновление аркадной мини-игры
    if (currentScreen == SCR_GAME && gameActive) {
      updateMiniGame();
    }
    // Анимация потока помпы на главном экране
    if (currentScreen == SCR_HOME) {
      animatePumpFlow();
    }
  }

  // Автоматическое отключение ручного полива питомца через 5 секунд
  if (pumpManualOffAt > 0 && now >= pumpManualOffAt) {
    pumpManualOffAt = 0;
    pumpState = false;
    lastPumpState = pumpState;
    if (!isSleeping) {
      if (currentScreen == SCR_HOME) refreshHomeValues();
      else if (currentScreen == SCR_SPROUT) drawSproutScreen();
    }
  }

  // Умный таймер полива (Auto Timer Mode)
  if (pumpAutoMode) {
    if (pumpState && (now - pumpTimerMark >= 15UL * 60UL * 1000UL)) {
      pumpState = false;
      pumpTimerMark = now;
      if (!isSleeping) refreshHomeValues();
    } else if (!pumpState && (now - pumpTimerMark >= 45UL * 60UL * 1000UL)) {
      pumpState = true;
      pumpTimerMark = now;
      if (!isSleeping) refreshHomeValues();
    }
  }

  // Опрос датчиков каждые 2 секунды
  if (now - lastSensorPoll >= 2000) {
    lastSensorPoll = now;

    // Режим Demo Waves: симулирует реалистичные синусоидальные колебания
    if (demoWavesEnabled) {
      float t = (float)now / 8000.0f;
      current_pH        = 6.2f + 0.35f * sinf(t);
      current_TDS       = 820 + (int)(60.0f * cosf(t * 0.8f));
      current_waterTemp = 21.8f + 1.2f * sinf(t * 0.5f);
      current_airTemp   = 23.5f + 2.0f * sinf(t * 0.4f);
      current_humidity  = 54.0f + 8.0f * cosf(t * 0.6f);
    }

    current_VPD = calculateVPD(current_airTemp, current_humidity);

    // Запись новых точек в циклическую историю
    appendMetricSample(METRIC_PH, current_pH);
    appendMetricSample(METRIC_TDS, (float)current_TDS);
    appendMetricSample(METRIC_WATER_TEMP, current_waterTemp);
    appendMetricSample(METRIC_AIR_TEMP, current_airTemp);
    appendMetricSample(METRIC_HUMIDITY, current_humidity);
    appendMetricSample(METRIC_VPD, current_VPD);

    if (!isSleeping) {
      if (currentScreen == SCR_HOME) {
        refreshHomeValues();
      } else if (currentScreen == SCR_DETAIL) {
        drawDetailScreen();
      }
    }
  }

  // Пассивное начисление опыта питомцу за идеальные условия экосистемы
  static unsigned long lastPassiveXp = 0;
  if (now - lastPassiveXp >= 60000UL) {
    lastPassiveXp = now;
    if (calculatePlantHealthScore() >= 80) {
      addSproutXP(2);
      if (currentScreen == SCR_SPROUT && !isSleeping) drawSproutScreen();
    }
  }

  delay(15);
}