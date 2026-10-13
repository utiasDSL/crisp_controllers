# Production headers must not depend on Eigen or libstdc++ implementation headers.
file(GLOB_RECURSE production_sources
  "${SOURCE_DIR}/src/*.cpp"
  "${SOURCE_DIR}/include/*.hpp"
  "${SOURCE_DIR}/include/*.h"
)
foreach(source IN LISTS production_sources)
  file(STRINGS "${source}" includes REGEX "^[ \t]*#[ \t]*include")
  foreach(include IN LISTS includes)
    if(include MATCHES "[<\"](Eigen/src/|binders[.]h[>\"]|backward/|bits/)")
      message(FATAL_ERROR "Private implementation header in ${source}: ${include}")
    endif()
  endforeach()
endforeach()
