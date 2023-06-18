#include "win.h"
win win::Instance;


win::win(void) : W(1280), H(736)
{
    char* v[1] = { 0 };
    int c = 1;

    glutInit(&c, v);

    //  glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB);
    glutInitWindowPosition(10, 10);
    glutInitWindowSize(W, H);
    glutCreateWindow("Feels bad man");
    glutDisplayFunc(Display);
    glutKeyboardFunc(Keyboard);

    glGenTextures(1, &screenBuf);
    glBindTexture(GL_TEXTURE_2D, screenBuf);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_FLOAT, nullptr);

    glGenFramebuffers(1, &fboId);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fboId);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D, screenBuf, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);  // if not already bound

    auto e = cudaGLSetGLDevice(0);
    e = cudaGraphicsGLRegisterImage(&screenRes, screenBuf, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);

    partMgr.Init();
}


void win::Display(void)
{
    static clock_t startTime = clock();
    clock_t endTime;
    glClearColor(1.0f * rand() / RAND_MAX, 1.0f * rand() / RAND_MAX, 1.0f * rand() / RAND_MAX, 1.0f * rand() / RAND_MAX);
    glClear(GL_COLOR_BUFFER_BIT);

    auto e = cudaGraphicsMapResources(1, &Instance.screenRes);
    cudaArray_t writeArray;
    e = cudaGraphicsSubResourceGetMappedArray(&writeArray, Instance.screenRes, 0, 0);
    cudaResourceDesc wdsc;
    wdsc.resType = cudaResourceTypeArray;
    wdsc.res.array.array = writeArray;
    cudaSurfaceObject_t writeSurface;
    e = cudaCreateSurfaceObject(&writeSurface, &wdsc);

    // clear background
    dim3 thread(32, 32);
    dim3 texDim(Instance.W, Instance.H);
    dim3 block(texDim.x / thread.x, texDim.y / thread.y);
    Fill << < block, thread >> > (writeSurface, texDim);

    // computations
    endTime = clock();
    double delta = (endTime - startTime) * 1000 / CLOCKS_PER_SEC; // delta time in miliseconds
    Instance.partMgr.Compute(writeSurface, texDim, delta);
    startTime = endTime;

    e = cudaDestroySurfaceObject(writeSurface);
    e = cudaGraphicsUnmapResources(1, &Instance.screenRes);
    e = cudaStreamSynchronize(0);

    glBlitFramebuffer(0, 0, Instance.W, Instance.H, 0, 0, Instance.W, Instance.H,
        GL_COLOR_BUFFER_BIT, GL_NEAREST);

    glFinish();
    glutSwapBuffers();
    glutPostRedisplay();


}

void win::Keyboard(unsigned char Key, int x, int y)
{
    if (Key == 27)
        exit(0);
    if (Key == 'F' || Key == 'f')
        glutFullScreenToggle();
}

void win::Run(void)
{
    glutMainLoop();
}

win::~win(void)
{
    partMgr.Kill();

    cudaError_t cudaStatus = cudaSuccess;
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaDeviceReset failed!");
}
