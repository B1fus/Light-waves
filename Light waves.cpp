// Physics dots.cpp : Определяет точку входа для приложения.
//
#pragma warning(disable : 4996)

#include "framework.h"
#include "Light waves.h"
#include "Globals.h"

#define MAX_LOADSTRING 100

// Глобальные переменные:
HINSTANCE hInst;                                // текущий экземпляр
WCHAR szTitle[MAX_LOADSTRING];                  // Текст строки заголовка
WCHAR szWindowClass[MAX_LOADSTRING];            // имя класса главного окна

// Отправить объявления функций, включенных в этот модуль кода:
ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK    About(HWND, UINT, WPARAM, LPARAM);

Mouse mouse;
Keyboard keyboard;

Light light(SCREEN_WIDTH, SCREEN_HEIGHT);

int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPWSTR    lpCmdLine,
    _In_ int       nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    // TODO: Разместите код здесь.
    Gdiplus::GdiplusStartupInput gdiplusStartupInput;
    ULONG_PTR gdiplusToken;
    Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, nullptr);



    // Инициализация глобальных строк
    LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
    LoadStringW(hInstance, IDC_LIGHTWAVES, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);

    // Выполнить инициализацию приложения:
    if (!InitInstance(hInstance, nCmdShow))
    {
        return FALSE;
    }

    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_LIGHTWAVES));

    MSG msg;

    // Цикл основного сообщения:
    while (GetMessage(&msg, nullptr, 0, 0))
    {
        if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    Gdiplus::GdiplusShutdown(gdiplusToken);
    return (int)msg.wParam;
}



//
//  ФУНКЦИЯ: MyRegisterClass()
//
//  ЦЕЛЬ: Регистрирует класс окна.
//
ATOM MyRegisterClass(HINSTANCE hInstance)
{
    WNDCLASSEXW wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);

    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = WndProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = hInstance;
    wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDC_LIGHTWAVES));
    wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wcex.lpszMenuName = MAKEINTRESOURCEW(IDC_LIGHTWAVES);
    wcex.lpszClassName = szWindowClass;
    wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

    return RegisterClassExW(&wcex);
}

//
//   ФУНКЦИЯ: InitInstance(HINSTANCE, int)
//
//   ЦЕЛЬ: Сохраняет маркер экземпляра и создает главное окно
//
//   КОММЕНТАРИИ:
//
//        В этой функции маркер экземпляра сохраняется в глобальной переменной, а также
//        создается и выводится главное окно программы.
//
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
    hInst = hInstance; // Сохранить маркер экземпляра в глобальной переменной

    HWND hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
        0, 0, SCREEN_WIDTH + 16, SCREEN_HEIGHT + 59, nullptr, nullptr, hInstance, nullptr);

    if (!hWnd)
    {
        return FALSE;
    }

    ShowWindow(hWnd, nCmdShow);
    UpdateWindow(hWnd);

    return TRUE;
}

//
//  ФУНКЦИЯ: WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  ЦЕЛЬ: Обрабатывает сообщения в главном окне.
//
//  WM_COMMAND  - обработать меню приложения
//  WM_PAINT    - Отрисовка главного окна
//  WM_DESTROY  - отправить сообщение о выходе и вернуться
//
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    static bool hasFocus = 1;
    static int win_width, win_height;
    static PAINTSTRUCT ps;
    static HDC hdc = BeginPaint(hWnd, &ps);
    static HDC Memhdc = CreateCompatibleDC(hdc);
    static HBITMAP Membitmap;
    static RECT Client_Rect;
    static Gdiplus::SolidBrush whiteBrush(Gdiplus::Color(255, 255, 255));
    static Gdiplus::SolidBrush redBrush(Gdiplus::Color(255, 0, 0));
    static Gdiplus::FontFamily  fontFamily(L"Times New Roman");
    static Gdiplus::Font font(&fontFamily, 12);
    static Gdiplus::Pen greenPen(Gdiplus::Color(0, 255, 0), 2.);
    static std::chrono::steady_clock::time_point start_time;


    switch (message)
    {
    case WM_KILLFOCUS:
        hasFocus = 0;
        mouse.buttons = 0;
        keyboard.clear();
        break;
    case WM_SETFOCUS:
        hasFocus = 1;
        break;
    case WM_COMMAND:
    {
        int wmId = LOWORD(wParam);
        // Разобрать выбор в меню:
        switch (wmId)
        {
        case IDM_ABOUT:
            DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
            break;
        case IDM_EXIT:
            DestroyWindow(hWnd);
            break;
        default:
            return DefWindowProc(hWnd, message, wParam, lParam);
        }
    }
    break;
    case WM_ERASEBKGND:
        return TRUE;
    case WM_KEYDOWN:
    case WM_KEYUP:
    case WM_SYSKEYDOWN:
    case WM_SYSKEYUP: {
        if (hasFocus) {
            static bool keyIsDown, keyWasDown;
            keyIsDown = ((lParam & (1 << 31)) == 0);
            keyWasDown = ((lParam & (1 << 30)) != 0);
            if (keyIsDown != keyWasDown) {
                keyboard.keys[wParam] = keyIsDown;
                if (wParam == VK_ESCAPE) {
                    PostQuitMessage(1);
                }
            }
        }
        break;
    }
    case WM_MOUSEMOVE: {
        mouse.coords = { LOWORD(lParam), HIWORD(lParam) };
        break;
    }
    case WM_LBUTTONDOWN: mouse.buttons |= MOUSE_LEFT; break;
    case WM_LBUTTONUP: mouse.buttons &= ~MOUSE_LEFT; break;
    case WM_MBUTTONDOWN: mouse.buttons |= MOUSE_MIDDLE; break;
    case WM_MBUTTONUP: mouse.buttons &= ~MOUSE_MIDDLE; break;
    case WM_RBUTTONDOWN: mouse.buttons |= MOUSE_RIGHT; break;
    case WM_RBUTTONUP: mouse.buttons &= ~MOUSE_RIGHT; break;
    case WM_XBUTTONDOWN: {
        if (GET_XBUTTON_WPARAM(wParam) == XBUTTON1)
            mouse.buttons |= MOUSE_X1;
        else
            mouse.buttons |= MOUSE_X2;
        break;
    }
    case WM_XBUTTONUP: {
        if (GET_XBUTTON_WPARAM(wParam) == XBUTTON1)
            mouse.buttons &= ~MOUSE_X1;
        else
            mouse.buttons &= ~MOUSE_X2;
        break;
    }
    case WM_MOUSEWHEEL: {
        if (wParam & (1 << 30))
            mouse.wheel = -1;
        else
            mouse.wheel = 1;
        wchar_t buffer[200];
        swprintf(buffer, 200, L"%d %d %d\n", wParam, wParam & (1<<30), mouse.wheel);
        OutputDebugString(buffer);
        break;
    }
    case WM_PAINT:
    {
        GetClientRect(hWnd, &Client_Rect);
        win_width = Client_Rect.right - Client_Rect.left;
        win_height = Client_Rect.bottom + Client_Rect.left;
        hdc = BeginPaint(hWnd, &ps);
        Memhdc = CreateCompatibleDC(hdc);
        Membitmap = CreateCompatibleBitmap(hdc, win_width, win_height);
        SelectObject(Memhdc, Membitmap);
        //drawing code goes in here
        start_time = std::chrono::steady_clock::now();
        static int* img = new int[SCREEN_WIDTH*SCREEN_HEIGHT];
        //static int* img;
        //for (int i = 0; i < win_height*win_width; i++) {
        //    if (i > 10000 && i < 100000) {
        //        img[i] = 0xffffff;
        //    }
        //    else {
        //        //img[i] = 0;
        //    }
        //}
        //GetBitmapBits(Membitmap, win_width * win_height * 4, lptr);


        Gdiplus::Graphics gr(Memhdc);

        if(mouse.buttons & MOUSE_LEFT)
        light.set_pixel_mat_height(mouse.coords.x, mouse.coords.y, 0x111111);
        if (mouse.buttons & MOUSE_RIGHT) {
            light.set_pixel_weight(mouse.coords.x, mouse.coords.y, 0);
            gr.FillRectangle(&whiteBrush, 100, 100, 10, 5);
            GetBitmapBits(Membitmap, sizeof(img[0]) * SCREEN_HEIGHT * SCREEN_WIDTH, img);
            light.set_weights_from_array(img);
        }
        if (keyboard.keys[' '] == 0)
            //light.move();
        move_gpu(light, img);
        //gr.FillRectangle(&whiteBrush, 0, 0, win_width, win_height); //clear window
        //GetBitmapBits(Membitmap, 4 * win_height * win_width, img);
        //img[mouse.coords.y*win_width+mouse.coords.x] = 0xff0000;
        //light.get_mat_height(img);

        SetBitmapBits(Membitmap, sizeof(img[0]) * SCREEN_WIDTH * SCREEN_HEIGHT, img);
        //output to debug panel
        //wchar_t buffer[200];
        //swprintf(buffer, 200, L"22222 %d", 10);
        //OutputDebugString(buffer);

        DELTA_TIME = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        //Sleep(500); if(DELTA_TIME>500) DELTA_TIME -= 500;
        REAL_FPS = 1 / DELTA_TIME;
        
        gr.DrawString((wchar_t*)std::to_wstring(REAL_FPS).c_str(), 5, &font, Gdiplus::PointF(0, 0), &redBrush);
        BitBlt(hdc, 0, 0, win_width, win_height, Memhdc, 0, 0, SRCCOPY);
        DeleteObject(Membitmap);
        DeleteDC(Memhdc);
        DeleteDC(hdc);
        EndPaint(hWnd, &ps);
        InvalidateRect(hWnd, NULL, false);
    }
    break;
    case WM_DESTROY:
        PostQuitMessage(0);

        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

// Обработчик сообщений для окна "О программе".
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    UNREFERENCED_PARAMETER(lParam);
    switch (message)
    {
    case WM_INITDIALOG:
        return (INT_PTR)TRUE;

    case WM_COMMAND:
        if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
        {
            EndDialog(hDlg, LOWORD(wParam));
            return (INT_PTR)TRUE;
        }
        break;
    }
    return (INT_PTR)FALSE;
}
