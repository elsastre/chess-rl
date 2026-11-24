/*
  pantalla_led.ino: ojos del bot de ajedrez en el display LED RGB de 32x32 (ATmega328p a 16 MHz)

  Basado en el programa ejemplo del curso de Microprocesadores (Display32x32.ino): misma memoria de
  pantalla (1 byte por pixel, 3 bits de color) y el mismo refresco por interrupcion del Timer2.

  CAMBIO DE CONEXION (obligatorio): en el ejemplo, A B C D van en PD0..PD3, pero PD0 y PD1 son RX y TX
  de la UART (y en un Arduino Uno o Nano, la conexion USB). Si se usa la comunicacion serie, la UART se
  adueña de esos pines y las filas del display se mezclan. Por eso A B C D pasan a PC0..PC3
  (A0..A3 en un Arduino Uno o Nano):

    PB0..PB2 = RGB0 (mitad de arriba)   PB3..PB5 = RGB1 (mitad de abajo)       igual que antes
    PC0..PC3 = A B C D (direccion de fila)                                     ANTES PD0..PD3
    PD4 = OE (activo en bajo)   PD5 = CLK   PD6 = LE / STB                     igual que antes
    PD0 = RX, PD1 = TX: comunicacion serie con la PC, 9600 baudios, 8N1

  Colores: 1 verde, 2 rojo, 4 azul, 3 amarillo, 5 cian, 6 violeta, 7 blanco, 0 apagado.

  Protocolo: una orden por linea (terminada en \n; el \r se ignora). Responde "OK" o "ERR".
    E:<emocion>   pensando | confiado | ganando | preocupado | sorprendido | perdiendo
    M:<texto>     jugada a mostrar abajo, hasta 5 caracteres (por ejemplo e2-e4, e4xd5, O-O, e7e8D)
                  "M:" sin texto borra la jugada
  Caracteres que entiende la fuente: 0-9 a-h - x = D T A C R O . + # y espacio.
  Se puede probar sin la PC del bot con el Monitor Serie del IDE de Arduino (9600, "Nueva linea").

  Pantalla: ojos en las filas 2 a 17 (cada ojo 12x16, el derecho es el izquierdo espejado) y la jugada
  en las filas 23 a 29 (fuente de 5x7). Animaciones: los ojos parpadean cada ~3,5 s y, pensando, miran
  a los lados.
*/

#include <avr/io.h>
#include <avr/interrupt.h>
#include <avr/pgmspace.h>
#include <string.h>

#define COLS 32
#define ROWS 32

volatile uint8_t screen[ROWS * COLS];     // memoria de pantalla (1024 bytes)
volatile uint16_t ticks = 0;              // cuenta las interrupciones del Timer2 (2500 por segundo)

// ---------------------------------------------------------------------------
// Ojos: 16 filas de 12 pixeles (los 12 bits de abajo de cada palabra; el bit 11 es la columna 0).
// Dibujados para el ojo izquierdo; el derecho se dibuja espejado.

const uint16_t OJO_ABIERTO[16] PROGMEM = {      // pensando (y base del parpadeo)
  0b000111111000,
  0b011111111110,
  0b011111111110,
  0b111111111111,
  0b111111111111,
  0b111111111111,
  0b111111111111,
  0b111111111111,
  0b111111111111,
  0b111111111111,
  0b111111111111,
  0b111111111111,
  0b111111111111,
  0b011111111110,
  0b011111111110,
  0b000111111000,
};

const uint16_t OJO_CONFIADO[16] PROGMEM = {     // abierto con el parpado de abajo curvo (sonrie)
  0b000000000000,
  0b000000000000,
  0b000111111000,
  0b011111111110,
  0b011111111110,
  0b111111111111,
  0b111111111111,
  0b111111111111,
  0b111111111111,
  0b111111111111,
  0b111111111111,
  0b111110011111,
  0b111000000111,
  0b010000000010,
  0b000000000000,
  0b000000000000,
};

const uint16_t OJO_GANANDO[16] PROGMEM = {      // arcos "^ ^" (muy contento)
  0b000000000000,
  0b000000000000,
  0b000000000000,
  0b000000000000,
  0b000011110000,
  0b001111111100,
  0b011111111110,
  0b111110011111,
  0b111100001111,
  0b111000000111,
  0b110000000011,
  0b110000000011,
  0b000000000000,
  0b000000000000,
  0b000000000000,
  0b000000000000,
};

const uint16_t OJO_PREOCUPADO[16] PROGMEM = {   // ceja inclinada (alta del lado de adentro) y ojo chico
  0b000000000011,
  0b000000111111,
  0b000111111100,
  0b111111100000,
  0b111000000000,
  0b000000000000,
  0b000000000000,
  0b000111111000,
  0b001111111100,
  0b011111111110,
  0b011111111110,
  0b011111111110,
  0b011111111110,
  0b001111111100,
  0b000111111000,
  0b000000000000,
};

const uint16_t OJO_SORPRENDIDO[16] PROGMEM = {  // aro grande con pupila chica
  0b000111111000,
  0b011111111110,
  0b011100001110,
  0b111000000111,
  0b110000000011,
  0b110000000011,
  0b110001100011,
  0b110011110011,
  0b110011110011,
  0b110001100011,
  0b110000000011,
  0b110000000011,
  0b111000000111,
  0b011100001110,
  0b011111111110,
  0b000111111000,
};

const uint16_t OJO_PERDIENDO[16] PROGMEM = {    // entrecerrado, parpado de arriba caido
  0b000000000000,
  0b000000000000,
  0b000000000000,
  0b000000000000,
  0b000000000000,
  0b000000000000,
  0b000000000000,
  0b111111111111,
  0b111111111111,
  0b011111111110,
  0b011111111110,
  0b001111111100,
  0b000111111000,
  0b000000000000,
  0b000000000000,
  0b000000000000,
};

const uint16_t OJO_CERRADO[16] PROGMEM = {      // parpadeo
  0, 0, 0, 0, 0, 0, 0,
  0b011111111110,
  0b111111111111,
  0b011111111110,
  0, 0, 0, 0, 0, 0,
};

// Emociones: mismo orden que NOMBRES. Color y si parpadea.
enum { PENSANDO, CONFIADO, GANANDO, PREOCUPADO, SORPRENDIDO, PERDIENDO, N_EMOCIONES };
const char *const NOMBRES[N_EMOCIONES] = {
  "pensando", "confiado", "ganando", "preocupado", "sorprendido", "perdiendo"
};
const uint16_t *const FORMA[N_EMOCIONES] = {
  OJO_ABIERTO, OJO_CONFIADO, OJO_GANANDO, OJO_PREOCUPADO, OJO_SORPRENDIDO, OJO_PERDIENDO
};
const uint8_t COLOR[N_EMOCIONES]     = { 7, 5, 1, 3, 6, 4 };
const uint8_t PARPADEA[N_EMOCIONES]  = { 1, 1, 0, 1, 1, 0 };
#define COLOR_JUGADA 7

#define OJO_FILA   2      // fila de arriba de los ojos
#define OJO_IZQ    3      // columna de arriba a la izquierda del ojo izquierdo
#define OJO_DER   17      // idem ojo derecho
#define TEXTO_FILA 23     // fila de arriba de la jugada

// ---------------------------------------------------------------------------
// Fuente de 5x7: 7 filas por caracter, los 5 bits de abajo (el bit 4 es la columna izquierda).
const char FUENTE_CHARS[] = "0123456789abcdefgh-x=DTACRO.+# ";
const uint8_t FUENTE[][7] PROGMEM = {
  {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}, // 0
  {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}, // 1
  {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F}, // 2
  {0x1F,0x02,0x04,0x02,0x01,0x11,0x0E}, // 3
  {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}, // 4
  {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}, // 5
  {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}, // 6
  {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}, // 7
  {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}, // 8
  {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}, // 9
  {0x00,0x00,0x0E,0x01,0x0F,0x11,0x0F}, // a
  {0x10,0x10,0x16,0x19,0x11,0x11,0x1E}, // b
  {0x00,0x00,0x0E,0x10,0x10,0x11,0x0E}, // c
  {0x01,0x01,0x0D,0x13,0x11,0x11,0x0F}, // d
  {0x00,0x00,0x0E,0x11,0x1F,0x10,0x0E}, // e
  {0x06,0x09,0x08,0x1C,0x08,0x08,0x08}, // f
  {0x00,0x0F,0x11,0x11,0x0F,0x01,0x0E}, // g
  {0x10,0x10,0x16,0x19,0x11,0x11,0x11}, // h
  {0x00,0x00,0x00,0x1F,0x00,0x00,0x00}, // -
  {0x00,0x00,0x11,0x0A,0x04,0x0A,0x11}, // x
  {0x00,0x00,0x1F,0x00,0x1F,0x00,0x00}, // =
  {0x1C,0x12,0x11,0x11,0x11,0x12,0x1C}, // D
  {0x1F,0x04,0x04,0x04,0x04,0x04,0x04}, // T
  {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}, // A
  {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}, // C
  {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}, // R
  {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}, // O
  {0x00,0x00,0x00,0x00,0x00,0x0C,0x0C}, // .
  {0x00,0x04,0x04,0x1F,0x04,0x04,0x00}, // +
  {0x0A,0x0A,0x1F,0x0A,0x1F,0x0A,0x0A}, // #
  {0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // espacio
};

// ---------------------------------------------------------------------------
// Refresco: cada interrupcion saca un par de filas (la fila n de la mitad de arriba y la de abajo).
ISR(TIMER2_COMPA_vect) {
  static uint8_t row = 0;
  const volatile uint8_t *top    = &screen[row * COLS];
  const volatile uint8_t *bottom = &screen[512 + row * COLS];
  uint8_t keepB = PORTB & 0xC0;                        // no tocar PB6, PB7 (cristal)

  PORTD &= ~_BV(PD6);                                  // LE = 0
  for (uint8_t x = 0; x < COLS; x++) {
    PORTD &= ~_BV(PD5);                                // CLK = 0
    PORTB = keepB | (top[x] & 0x07) | ((bottom[x] & 0x07) << 3);
    PORTD |= _BV(PD5);                                 // CLK = 1
  }
  PORTD |= _BV(PD4);                                   // OE = 1, display apagado
  PORTD |= _BV(PD6);                                   // LE = 1, se guarda la fila
  PORTC = (PORTC & 0xF0) | row;                        // direccion de fila en A B C D (PC0..PC3)
  PORTD &= ~(_BV(PD4) | _BV(PD5) | _BV(PD6));          // OE = 0 (encendido), CLK = 0, LE = 0

  row = (row + 1) & 0x0F;
  ticks++;
}

// ---------------------------------------------------------------------------
// Comunicacion serie (UART0 por registros, sin la clase Serial: asi compila igual con el IDE de
// Arduino o con avr-gcc solo). Lo recibido entra a un buffer circular desde la interrupcion.
#define RX_TAM 64
volatile uint8_t rx_buf[RX_TAM];
volatile uint8_t rx_cab = 0, rx_cola = 0;

ISR(USART_RX_vect) {
  uint8_t c = UDR0;
  uint8_t sig = (rx_cab + 1) % RX_TAM;
  if (sig != rx_cola) { rx_buf[rx_cab] = c; rx_cab = sig; }   // si se llena, se descarta
}

void uart_iniciar() {
  UBRR0H = 0;
  UBRR0L = 103;                                        // 16 MHz / (16 * 104) = 9600 baudios
  UCSR0A = 0;
  UCSR0B = _BV(RXEN0) | _BV(TXEN0) | _BV(RXCIE0);
  UCSR0C = _BV(UCSZ01) | _BV(UCSZ00);                  // 8 bits, sin paridad, 1 de parada
}

int uart_leer() {                                      // -1 si no llego nada
  if (rx_cab == rx_cola) return -1;
  uint8_t c = rx_buf[rx_cola];
  rx_cola = (rx_cola + 1) % RX_TAM;
  return c;
}

void uart_escribir(const char *s) {
  while (*s) {
    while (!(UCSR0A & _BV(UDRE0))) {}
    UDR0 = *s++;
  }
}

// ---------------------------------------------------------------------------
// Dibujo
void borrar_zona(uint8_t fila0, uint8_t fila1) {       // filas fila0..fila1-1
  for (uint16_t i = fila0 * COLS; i < fila1 * COLS; i++) screen[i] = 0;
}

void pixel(int8_t fila, int8_t col, uint8_t color) {
  if (fila >= 0 && fila < ROWS && col >= 0 && col < COLS) screen[fila * COLS + col] = color;
}

void dibujar_ojo(const uint16_t *forma, int8_t fila, int8_t col, uint8_t espejo, uint8_t color) {
  for (uint8_t y = 0; y < 16; y++) {
    uint16_t bits = pgm_read_word(&forma[y]);
    for (uint8_t x = 0; x < 12; x++) {
      uint8_t b = espejo ? x : 11 - x;
      if (bits & (1 << b)) pixel(fila + y, col + x, color);
    }
  }
}

void dibujar_texto(const char *t, uint8_t fila, uint8_t color) {
  uint8_t n = strlen(t);
  int8_t col = (COLS - (n * 6 - 1)) / 2;               // centrado (5 de ancho + 1 de espacio)
  for (uint8_t i = 0; i < n; i++, col += 6) {
    const char *p = strchr(FUENTE_CHARS, t[i]);
    if (!p || !t[i]) continue;
    uint8_t k = p - FUENTE_CHARS;
    for (uint8_t y = 0; y < 7; y++) {
      uint8_t bits = pgm_read_byte(&FUENTE[k][y]);
      for (uint8_t x = 0; x < 5; x++)
        if (bits & (0x10 >> x)) pixel(fila + y, col + x, color);
    }
  }
}

// ---------------------------------------------------------------------------
// Estado
uint8_t emocion = PENSANDO;
char jugada[6] = "";
uint8_t redibujar = 1;
uint8_t ultimo_cuadro = 255;

uint16_t leer_ticks() {
  uint8_t s = SREG;
  cli();
  uint16_t t = ticks;
  SREG = s;
  return t;
}

// Cuadro de animacion segun el tiempo: 0 normal, 1 parpadeo; pensando suma la mirada (2..5).
uint8_t cuadro_actual() {
  uint16_t t = leer_ticks();
  uint8_t c = 0;
  if (PARPADEA[emocion] && (t % 8750) < 375) c = 1;    // parpadeo de 0,15 s cada 3,5 s
  if (emocion == PENSANDO) c += 2 * (1 + (t / 1500) % 4);  // cambia la mirada cada 0,6 s
  return c;
}

void dibujar_todo(uint8_t cuadro) {
  const uint16_t *forma = (cuadro & 1) ? OJO_CERRADO : FORMA[emocion];
  int8_t dx = 0;
  if (emocion == PENSANDO) {
    static const int8_t MIRADA[4] = { -2, 0, 2, 0 };
    dx = MIRADA[(cuadro >> 1) - 1];
  }
  borrar_zona(0, 20);
  dibujar_ojo(forma, OJO_FILA, OJO_IZQ + dx, 0, COLOR[emocion]);
  dibujar_ojo(forma, OJO_FILA, OJO_DER + dx, 1, COLOR[emocion]);
  if (redibujar) {                                     // la jugada solo cambia con una orden
    borrar_zona(20, 32);
    dibujar_texto(jugada, TEXTO_FILA, COLOR_JUGADA);
  }
}

// Procesa una linea recibida. Devuelve 1 si la entendio.
uint8_t procesar(char *linea) {
  if (linea[0] == 'E' && linea[1] == ':') {
    for (uint8_t i = 0; i < N_EMOCIONES; i++)
      if (strcmp(linea + 2, NOMBRES[i]) == 0) { emocion = i; return 1; }
    return 0;
  }
  if (linea[0] == 'M' && linea[1] == ':') {
    if (strlen(linea + 2) > 5) return 0;
    strcpy(jugada, linea + 2);
    redibujar = 1;
    return 1;
  }
  return 0;
}

// ---------------------------------------------------------------------------
void setup() {
  DDRB |= 0b00111111;                                  // PB0..PB5 salidas (RGB0, RGB1), apagadas
  PORTB &= 0b11000000;
  DDRC = 0x0F;                                         // PC0..PC3 = A B C D, salidas
  PORTC = 0x00;
  DDRD = (DDRD & 0x03) | 0b01110000;                   // PD4 OE, PD5 CLK, PD6 LE; PD0/PD1 de la UART
  PORTD &= 0x8F;

  cli();
  TCCR2A = _BV(WGM21);                                 // Timer2: CTC, prescaler 256, OCR2A = 24
  TCCR2B = _BV(CS22) | _BV(CS21);                      //   16 MHz / 256 / 25 = 2500 Hz
  OCR2A  = 24;
  TIMSK2 = _BV(OCIE2A);
  uart_iniciar();
  sei();

  borrar_zona(0, 32);
  uart_escribir("LISTO\n");
}

void loop() {
  static char linea[24];
  static uint8_t largo = 0;
  int c;
  while ((c = uart_leer()) >= 0) {
    if (c == '\r') continue;
    if (c == '\n') {
      linea[largo] = 0;
      if (largo > 0) uart_escribir(procesar(linea) ? "OK\n" : "ERR\n");
      largo = 0;
      ultimo_cuadro = 255;                             // forzar redibujo
    } else if (largo < sizeof(linea) - 1) {
      linea[largo++] = (char)c;
    }
  }
  uint8_t cuadro = cuadro_actual();
  if (cuadro != ultimo_cuadro) {
    dibujar_todo(cuadro);
    ultimo_cuadro = cuadro;
    redibujar = 0;
  }
}

#ifndef ARDUINO
// Para compilar con avr-gcc solo (sin el IDE de Arduino)
int main(void) {
  setup();
  for (;;) loop();
}
#endif
