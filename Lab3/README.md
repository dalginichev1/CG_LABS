# Лабораторная работа №3: Текстурирование

Структура проекта:

    source/        main.cpp, application.*, graphics_internal.*
    shaders/       shader.vert/frag (освещение + текстуры), unlit.vert/frag (фигуры источников)
    textures/      <имя>_albedo.png, <имя>_specular.png, <имя>_emissive.png
    tools/         generate_textures.py - генератор процедурных текстур (numpy + pillow)

## Запуск

1. Скомпилировать шейдеры (**обязательно**, шейдеры изменились): `compile_shaders.bat`
   (или `compile_shaders.sh`). Старые `*.spv` нужно перезаписать.
2. Собрать проект через CMake (stb_image подтягивается через FetchContent).
3. Запускать из каталога проекта (как и шейдеры, каталог `textures/` ищется
   относительно папки, где лежит `shaders/`).

## Как добавить свою текстуру

Положите в `textures/` файл `<имя>_albedo.<png|jpg|bmp|tga|gif>`; по желанию -
`<имя>_specular.*` и `<имя>_emissive.*`. Набор `<имя>` появится в выпадающем
списке у каждого объекта. Если specular/emissive нет, используется служебная
чёрная текстура 1x1 (блеска/свечения не будет).
