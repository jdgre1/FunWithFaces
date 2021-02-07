
#include <fstream>
using namespace std;


void RunFaceApp();

void printCWD(char* argv[]);
int main(int argc, char* argv[])
{
    printCWD(argv);
    RunFaceApp();

}



void printCWD(char* argv[]) {
    char the_path[256];

    getcwd(the_path, 255);
    strcat(the_path, "/");
    strcat(the_path, argv[0]);

    printf("%s\n", the_path);

}