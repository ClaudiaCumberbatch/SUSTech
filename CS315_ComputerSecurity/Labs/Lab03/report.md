# Lab 03 Report
Name: Sicheng Zhou
SID: 12110644

## Task 01 Crashing the Program

A string of "%s" will move va_list pointer to somewhere with invalid address, leading to program crash.

![Crashing the Program](image.png)

## Task 02 Printing Out the Server Program’s Memory

### 2A Stack Data

I need 36 %x to print out the first four bytes of my input.

![Stack Data](image-1.png)

### 2B Heap Data

At first I was trying to construct an input string with sceret message address at the beginning, followed with several "%x" and a "%s" to move `va_list` to the correct place then print the message out as a string. However, as a Mac user, I can only use 64 bit system and my target address is `0x0000000000458248`. As stated in Task 5, when printf() parses the format string, it will stop the parsing when it sees a zero, so I have to use `k$` to move the pointer. Here is my code to generate the bad file and the final result.

![Heap Data Code](image-2.png)

![Heap Data Result](image-3.png)

## Task 03 Modifying the Server Program’s Memory

### 3A Change the value to a different value

First use `s = "%38$.16x"` to print out the address, then change the "x" into "n" to modify the content in this piece of memory. Because this "s" is put at the beginning and so far no character has been output, the last four bytes of the target value are changed into 0.

![3A Code](image-4.png)

![3A Result](image-5.png)

### 3B Change the value to 0x5000

1. `"%47$.16n" + "%48$.16n"` make the target value = 0;
2. `"..%.77x."` print out 80 characters;
3. `"%51$.16n"` set that address to 50.

![3B Code](image-6.png)

![3B Result](image-7.png)

### 3C Change the value to 0xAABBCCDD

`%hhn` only modify one byte.

![3C Code](image-8.png)

![3C Result](image-9.png)

## Task 04 Inject Malicious Code into the Server Program

### Answer Questions

**Question 1: What are the memory addresses at the locations marked 2 by and 3?**

address 2 = frame pointer + `0x8`, in my case `0x0000ffffffffef60 + 0x8 = 0x0000ffffffffef68`

3 is the beginning of the buffer so it's `0x0000fffffffff038`.

**Question 2: How many %x format specifiers do we need to move the format string argument pointer to 3? Remember, the argument pointer starts from the location above 1.**

By trail and error, `%50$.16x` can access the first piece of content after `s`.

### 4A Shell

![4A Code](image-10.png)

![4A Result](image-11.png)

### 4B Getting a Reverse Shell

Change line 23 into:

![4B Code](image-12.png)

Listen to prot 9091, then got connection.

![4B Result1](image-13.png)

![4B Result2](image-14.png)


## Task 05 Attacking the 64-bit Server Program

Already finished.

## Task 06 Fixing the Problem

The warning means that the string format is not a constant, and there are no parameters to format the string.

Change `printf(msg)` into `printf("%s", msg)`, then the warning disappeared.

![6 Code](image-15.png)

The attack failed.

![6 Result](image-16.png)