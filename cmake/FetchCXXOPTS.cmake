include_guard(GLOBAL)

# Configure cxxopts dependency handling.
# Exposes:
#  - CXXOPTS_INCLUDE_DIR (PATH, optional): user-provided include path
#  - CXXOPTS_IMPORTED_INCLUDE: resolved include path to use in targets

if(NOT DEFINED CXXOPTS_INCLUDE_DIR)
  set(CXXOPTS_INCLUDE_DIR "" CACHE PATH "Path to cxxopts include directory (optional)")
endif()

unset(CXXOPTS_IMPORTED_INCLUDE CACHE)

if(NOT CXXOPTS_INCLUDE_DIR)
  include(FetchContent)
  set(FETCHCONTENT_QUIET OFF)
  FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.1.1
  )
  FetchContent_GetProperties(cxxopts)
  if(NOT cxxopts_POPULATED)
    FetchContent_Populate(cxxopts)
  endif()
  set(CXXOPTS_IMPORTED_INCLUDE "${cxxopts_SOURCE_DIR}/include" CACHE PATH "Resolved cxxopts include path" FORCE)
  message(STATUS "Fetched cxxopts from ${cxxopts_SOURCE_DIR}")
elseif(CXXOPTS_INCLUDE_DIR)
  set(CXXOPTS_IMPORTED_INCLUDE "${CXXOPTS_INCLUDE_DIR}" CACHE PATH "Resolved cxxopts include path" FORCE)
  message(STATUS "Using cxxopts from ${CXXOPTS_IMPORTED_INCLUDE}")
else()
  message(STATUS "cxxopts not configured. Set CXXOPTS_INCLUDE_DIR or enable DOWNLOAD_CXXOPTS.")
endif()

