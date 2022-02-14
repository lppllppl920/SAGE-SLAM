
#ifndef CAMERADRIVERS_EXPORT_H
#define CAMERADRIVERS_EXPORT_H

#ifdef CAMERADRIVERS_STATIC_DEFINE
#  define CAMERADRIVERS_EXPORT
#  define CAMERADRIVERS_NO_EXPORT
#else
#  ifndef CAMERADRIVERS_EXPORT
#    ifdef CameraDrivers_EXPORTS
        /* We are building this library */
#      define CAMERADRIVERS_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define CAMERADRIVERS_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef CAMERADRIVERS_NO_EXPORT
#    define CAMERADRIVERS_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef CAMERADRIVERS_DEPRECATED
#  define CAMERADRIVERS_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef CAMERADRIVERS_DEPRECATED_EXPORT
#  define CAMERADRIVERS_DEPRECATED_EXPORT CAMERADRIVERS_EXPORT CAMERADRIVERS_DEPRECATED
#endif

#ifndef CAMERADRIVERS_DEPRECATED_NO_EXPORT
#  define CAMERADRIVERS_DEPRECATED_NO_EXPORT CAMERADRIVERS_NO_EXPORT CAMERADRIVERS_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef CAMERADRIVERS_NO_DEPRECATED
#    define CAMERADRIVERS_NO_DEPRECATED
#  endif
#endif

#endif /* CAMERADRIVERS_EXPORT_H */
