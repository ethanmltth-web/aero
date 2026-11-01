# Aero

Aero is a Mac app that opens a 3D model and shows a wind-tunnel preview: smoke lines, a vector field, streamlines, and pressure colors.

It is a fast preview, not a certified wind-tunnel test. The longest side of a model is scaled to 1 meter before the force numbers are estimated.

## Put it in Applications

On this Mac, double-click `Install to Applications.command`. Aero then appears in the Applications folder and in Launchpad.

The first time you open the installed app on a new Mac, it downloads Python and the graphics libraries into `~/Library/Application Support/Aero`. That step needs a network connection and can take a few minutes.

## Open a model

Use **Open 3D model**, or drop a file on the window. STL, OBJ, PLY, GLB, and other common 3D files work. **Samples** includes a wing, sphere, brick, cylinder, and cone.

Drag in the view to look around, scroll to zoom, and right-drag to pan. Wind speed, angle, wake, smoke, and the vector slice are in the side panels.

## Run it from this folder

Double-click `Setup Aero.command` once, then double-click `Aero.app` or `Run Aero.command`.
