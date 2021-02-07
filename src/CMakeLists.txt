

### ~~~~~~~~~~~~ BUILD RELEVANT LIBRARIES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

add_library(FaceLibrary STATIC face_library.cpp ../include/face_library.h)
target_link_libraries(FaceLibrary ${OpenCV_LIBS})



### ~~~~~~~~~~~~ CREATE EXECUTABLE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

set(project_sources main.cpp FacesAppRun.cpp)
add_executable(${CMAKE_PROJECT_NAME} ${project_sources} )
target_link_libraries( ${CMAKE_PROJECT_NAME} FaceLibrary ) 
target_precompile_headers(${CMAKE_PROJECT_NAME} PRIVATE ../include/pch.h)



