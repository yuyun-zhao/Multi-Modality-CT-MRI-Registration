
cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

PROJECT(Registration)

FIND_PACKAGE(ITK REQUIRED )
FIND_PACKAGE(VTK REQUIRED )

IF(ITK_FOUND)
INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
MESSAGE(FATAL_ERROR
"ITK not found. Please set ITK_DIR.")
ENDIF(ITK_FOUND)


if (NOT VTK_FOUND)
  message("Skipping (Registration: ${VTK_NOT_FOUND_MESSAGE}")
  return ()
endif()
message (STATUS "VTK_VERSION: ${VTK_VERSION}")

include(${VTK_USE_FILE})
 #  target_link_libraries(Registration PRIVATE ${VTK_LIBRARIES})

ADD_EXECUTABLE(Registration main.cxx)
TARGET_LINK_LIBRARIES(Registration ${VTK_LIBRARIES} ${ITK_LIBRARIES})