# 2023-05-03
# https://stackoverflow.com/questions/69313876/how-to-get-points-of-the-svg-paths
# 实现了一个均匀分配点的算法, 但后来发现 path.length() 太耗时了, 故废弃此算法.

from svg.path import parse_path
import svg.path.path
from xml.dom import minidom
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

SVG_STRING =\
"""
<svg xmlns="http://www.w3.org/2000/svg">
  <g>
    <path class="cls-1" d="M1.52,394.36H22.32l8.9-35.35H10.55l1.52-7.54h21.04l11-43.7h11.5l-11,43.7h20.54l11-43.7h11.51l-11,43.7h20.42l-1.81,7.54h-20.51l-8.9,35.35h20.66l-2.09,7.54h-20.47l-11,43.7h-11.5l11-43.7H31.92l-11,43.7H9.41l11.01-43.7H0l1.52-7.54Zm52.84,0l8.9-35.35h-20.54l-8.9,35.35h20.54Z"/>
    <path class="cls-1" d="M101.17,442.75c6.78-.68,11.06-1.92,12.84-3.72,1.77-1.8,2.66-6.67,2.66-14.62v-48.59c0-4.41-.29-7.47-.86-9.17-.95-2.78-2.95-4.18-5.99-4.18-.7,0-1.38,.07-2.04,.2-.67,.14-2.58,.68-5.75,1.63v-3.16l4.09-1.43c11.09-3.87,18.83-6.79,23.2-8.76,1.77-.81,2.92-1.22,3.42-1.22,.13,.48,.19,.98,.19,1.53v73.14c0,7.74,.87,12.6,2.62,14.57,1.74,1.97,5.69,3.23,11.84,3.77v2.85h-46.21v-2.85Zm15.12-136.21c1.9-2.09,4.25-3.14,7.04-3.14s5.05,1.03,6.99,3.09c1.93,2.06,2.9,4.58,2.9,7.55s-.97,5.39-2.9,7.45c-1.93,2.06-4.26,3.09-6.99,3.09s-5.13-1.03-7.04-3.09c-1.9-2.06-2.85-4.55-2.85-7.45s.95-5.41,2.85-7.5Z"/>
    <path class="cls-1" d="M154.99,442.75c4.5-.61,7.65-1.82,9.46-3.62,1.81-1.8,2.71-5.82,2.71-12.07v-51.75c0-4.31-.38-7.34-1.14-9.09-1.2-2.56-3.68-3.84-7.42-3.84-.57,0-1.16,.04-1.76,.1-.6,.07-1.35,.17-2.23,.31v-3.57c2.6-.81,8.75-2.92,18.45-6.32l8.94-3.16c.44,0,.71,.17,.81,.51,.1,.34,.14,.81,.14,1.43v14.97c5.82-5.84,10.38-9.85,13.67-12.02,4.94-3.33,10.06-4.99,15.38-4.99,4.3,0,8.23,1.32,11.77,3.97,6.83,5.16,10.25,14.36,10.25,27.6v47.57c0,4.89,.91,8.42,2.74,10.59,1.82,2.17,4.88,3.29,9.15,3.36v2.85h-40.6v-2.85c4.63-.68,7.86-2.05,9.7-4.13,1.84-2.07,2.76-6.57,2.76-13.5v-43.5c0-5.84-1.01-10.68-3.03-14.52-2.02-3.84-5.74-5.76-11.16-5.76-3.72,0-7.5,1.36-11.35,4.07-2.14,1.56-4.92,4.14-8.32,7.74v57.04c0,4.89,1.01,8.17,3.04,9.83,2.03,1.67,5.23,2.56,9.6,2.7v2.85h-41.55v-2.85Z"/>
    <path class="cls-1" d="M317.74,356.09c5.74,4.37,8.61,9.15,8.61,14.36,0,2.23-.73,4.31-2.19,6.24-1.46,1.93-3.71,2.89-6.75,2.89-2.22,0-4.17-.81-5.85-2.44-1.68-1.63-2.9-4.01-3.66-7.13l-1.14-4.89c-.82-3.6-2.25-6.15-4.28-7.64-2.09-1.43-4.85-2.14-8.27-2.14-7.23,0-13.33,3.38-18.3,10.15-4.98,6.77-7.46,15.76-7.46,26.98,0,10.27,2.74,19.48,8.23,27.64,5.48,8.16,12.88,12.24,22.2,12.24,6.59,0,12.46-2.31,17.59-6.94,2.92-2.65,6.21-6.87,9.89-12.65l2.66,1.73c-3.61,8.16-7.42,14.62-11.41,19.37-7.67,9.11-16.55,13.66-26.62,13.66s-18.64-4.18-26.24-12.53c-7.61-8.35-11.41-19.69-11.41-34.02s4.15-26.14,12.46-36.26c8.3-10.12,18.79-15.18,31.47-15.18,7.92,0,14.75,2.18,20.49,6.55Z"/>
    <path class="cls-1" d="M339.28,442.75c5.9-.61,9.89-1.88,11.98-3.82,2.09-1.94,3.14-5.72,3.14-11.36v-99.52c0-4.48-.35-7.57-1.05-9.27-1.27-2.85-3.84-4.28-7.7-4.28-.89,0-1.85,.1-2.9,.31-1.05,.2-2.36,.51-3.95,.92v-3.36c8.56-2.44,18.86-5.7,30.9-9.78,.44,0,.71,.2,.81,.61,.09,.41,.14,1.29,.14,2.65v122.13c0,5.91,.95,9.73,2.85,11.46,1.9,1.73,5.83,2.84,11.79,3.31v2.85h-46.02v-2.85Z"/>
    <path class="cls-1" d="M419.53,351.28v65.19c0,4.62,.6,8.32,1.81,11.1,2.34,5.16,6.66,7.74,12.93,7.74,4.31,0,8.53-1.53,12.65-4.58,2.34-1.7,4.72-4.04,7.13-7.03v-55.82c0-5.23-.95-8.66-2.85-10.29s-5.71-2.61-11.41-2.95v-3.36h30.71v71.71c0,4.62,.78,7.79,2.33,9.52,1.55,1.73,4.83,2.5,9.84,2.29v2.85c-3.49,1.02-6.05,1.78-7.7,2.29-1.65,.51-4.41,1.44-8.27,2.8-1.65,.61-5.26,2.07-10.84,4.38-.32,0-.51-.15-.57-.46-.06-.31-.1-.66-.1-1.07v-16.4c-4.31,5.5-8.24,9.58-11.79,12.22-5.39,4.07-11.09,6.11-17.12,6.11-5.52,0-10.71-2.11-15.59-6.32-4.94-4.15-7.42-11.12-7.42-20.91v-52.93c0-5.39-1.08-9-3.23-10.82-1.4-1.14-4.37-1.95-8.94-2.43v-2.85h28.43Z"/>
    <path class="cls-1" d="M545.62,352.6c2.53,1.43,4.98,3.4,7.32,5.91v-32.39c0-4.14-.43-6.99-1.28-8.56-.86-1.56-2.9-2.34-6.13-2.34-.76,0-1.43,.04-2,.1-.57,.07-1.84,.2-3.8,.41v-3.36l7.8-2.14c2.85-.81,5.71-1.66,8.56-2.55,2.85-.88,5.36-1.73,7.51-2.55,1.01-.34,2.69-.98,5.04-1.94l.57,.2-.19,10.7c-.06,3.87-.13,7.86-.19,11.97-.06,4.11-.1,8.17-.1,12.17l-.19,83.22c0,4.42,.51,7.5,1.52,9.27,1.01,1.77,3.71,2.65,8.08,2.65,.7,0,1.39-.02,2.09-.05,.7-.03,1.39-.12,2.09-.25v3.36c-.38,.14-4.98,1.83-13.79,5.09l-14.93,6.01-.67-.92v-12.53c-3.55,4.14-6.62,7.1-9.22,8.86-4.63,3.06-9.98,4.58-16.07,4.58-10.78,0-19.51-4.46-26.2-13.4-6.69-8.93-10.03-19.27-10.03-31.02,0-14.73,4.01-27.38,12.03-37.94,8.02-10.56,17.83-15.84,29.43-15.84,4.63,0,8.87,1.09,12.74,3.26Zm2.76,79.25c3.04-3.12,4.56-6.08,4.56-8.86v-43.7c0-8.83-2.2-15.06-6.61-18.69-4.41-3.63-8.7-5.45-12.88-5.45-7.99,0-14.2,3.79-18.64,11.36-4.44,7.57-6.66,16.89-6.66,27.96s2.36,20.68,7.08,29.23c4.72,8.56,11.77,12.83,21.16,12.83,4.94,0,8.94-1.56,11.98-4.69Z"/>
    <path class="cls-1" d="M652.21,359.99c6.72,6.62,10.08,16.01,10.08,28.17h-60.38c.63,15.72,3.96,27.17,9.98,34.35,6.02,7.18,13.15,10.77,21.4,10.77,6.66,0,12.27-1.86,16.83-5.59,4.56-3.73,8.78-9.01,12.65-15.86l3.33,1.22c-2.6,8.64-7.46,16.63-14.6,23.98-7.13,7.35-15.86,11.02-26.2,11.02-11.92,0-21.13-4.82-27.62-14.46-6.5-9.64-9.75-20.74-9.75-33.31,0-13.65,3.77-25.43,11.32-35.35,7.54-9.91,17.37-14.87,29.48-14.87,8.94,0,16.77,3.31,23.49,9.93Zm-45.55,7.79c-2.03,3.67-3.49,8.29-4.37,13.85h40.13c-.7-6.79-1.9-11.85-3.61-15.18-3.11-5.98-8.31-8.96-15.59-8.96s-12.74,3.43-16.55,10.29Z"/>
    <path class="cls-1" d="M822.9,431.84v15.45l-97.85-51.64v-5.6l97.85-51.64v15.48l-72.46,38.81,72.46,39.15Z"/>
    <path class="cls-1" d="M832.88,442.75c6.78-.68,11.06-1.92,12.84-3.72,1.77-1.8,2.66-6.67,2.66-14.62v-48.59c0-4.41-.29-7.47-.86-9.17-.95-2.78-2.95-4.18-5.99-4.18-.7,0-1.38,.07-2.04,.2-.67,.14-2.58,.68-5.75,1.63v-3.16l4.09-1.43c11.09-3.87,18.83-6.79,23.2-8.76,1.77-.81,2.92-1.22,3.42-1.22,.13,.48,.19,.98,.19,1.53v73.14c0,7.74,.87,12.6,2.62,14.57,1.74,1.97,5.69,3.23,11.84,3.77v2.85h-46.21v-2.85Zm15.12-136.21c1.9-2.09,4.25-3.14,7.04-3.14s5.05,1.03,6.99,3.09c1.93,2.06,2.9,4.58,2.9,7.55s-.97,5.39-2.9,7.45c-1.93,2.06-4.26,3.09-6.99,3.09s-5.13-1.03-7.04-3.09c-1.9-2.06-2.85-4.55-2.85-7.45s.95-5.41,2.85-7.5Z"/>
    <path class="cls-1" d="M900.16,364.06c8.02-9.34,18.33-14.01,30.95-14.01s22.95,4.42,31.19,13.24c8.24,8.83,12.36,20.58,12.36,35.24,0,13.51-3.99,25.26-11.98,35.24-7.99,9.98-18.32,14.97-31,14.97s-22.47-4.79-30.9-14.36c-8.43-9.58-12.65-21.59-12.65-36.06,0-13.51,4.01-24.94,12.03-34.28Zm15.18-2.6c-6.35,6.18-9.53,16.84-9.53,31.98,0,12.09,2.56,23.36,7.67,33.82,5.11,10.46,12.21,15.69,21.29,15.69,7.11,0,12.59-3.5,16.43-10.49,3.84-6.99,5.76-16.16,5.76-27.5s-2.45-22.82-7.34-33.21c-4.89-10.39-11.97-15.58-21.24-15.58-5.02,0-9.37,1.77-13.05,5.3Z"/>
    <path class="cls-1" d="M990.73,413.51h3.14c1.45,7.74,3.42,13.68,5.88,17.83,4.43,7.61,10.91,11.41,19.45,11.41,4.74,0,8.49-1.41,11.24-4.23,2.75-2.82,4.13-6.47,4.13-10.95,0-2.85-.79-5.6-2.38-8.25-1.58-2.65-4.37-5.23-8.37-7.74l-10.65-6.52c-7.8-4.48-13.53-9-17.21-13.55-3.68-4.55-5.52-9.91-5.52-16.09,0-7.6,2.53-13.85,7.61-18.74,5.07-4.89,11.44-7.33,19.11-7.33,3.36,0,7.05,.68,11.08,2.04,4.03,1.36,6.29,2.04,6.8,2.04,1.14,0,1.96-.17,2.47-.51,.51-.34,.95-.88,1.33-1.63h2.28l.67,28.42h-2.95c-1.27-6.59-2.98-11.71-5.13-15.38-3.93-6.79-9.6-10.19-17.02-10.19-4.44,0-7.92,1.46-10.46,4.38-2.54,2.92-3.8,6.35-3.8,10.29,0,6.25,4.37,11.82,13.12,16.71l12.55,7.23c13.5,7.88,20.25,17.05,20.25,27.5,0,8.01-2.8,14.57-8.4,19.66-5.6,5.09-12.93,7.64-21.97,7.64-3.8,0-8.1-.68-12.91-2.04-4.81-1.36-7.66-2.04-8.54-2.04-.76,0-1.42,.29-1.99,.87-.57,.58-1.01,1.27-1.33,2.09h-2.47v-32.9Z"/>
    <path class="cls-1" d="M1105.88,351.68v7.33h-19.4l-.19,58.67c0,5.16,.41,9.07,1.24,11.71,1.52,4.69,4.5,7.03,8.94,7.03,2.28,0,4.26-.58,5.94-1.73,1.68-1.15,3.6-2.99,5.75-5.5l2.47,2.24-2.09,3.06c-3.3,4.75-6.78,8.12-10.46,10.08-3.68,1.97-7.23,2.95-10.65,2.95-7.48,0-12.55-3.57-15.21-10.7-1.46-3.87-2.19-9.23-2.19-16.09v-61.73h-10.36c-.32-.2-.56-.41-.71-.61-.16-.2-.24-.47-.24-.81,0-.68,.14-1.2,.43-1.58,.29-.37,1.19-1.24,2.71-2.6,4.38-3.87,7.53-7.01,9.46-9.42,1.93-2.41,6.48-8.78,13.65-19.1,.82,0,1.31,.07,1.47,.2,.16,.14,.24,.65,.24,1.53v25.06h19.21Z"/>
    <path class="cls-1" d="M1111.3,442.24c5.83-.54,9.7-1.65,11.6-3.31,1.9-1.66,2.85-5.21,2.85-10.65v-45.94c0-6.72-.59-11.53-1.76-14.41-1.17-2.89-3.31-4.33-6.42-4.33-.63,0-1.47,.09-2.52,.25-1.05,.17-2.14,.39-3.28,.66v-3.36c3.6-1.36,7.3-2.75,11.09-4.18,3.79-1.43,6.41-2.44,7.87-3.06,3.16-1.29,6.41-2.75,9.76-4.38,.44,0,.71,.17,.81,.51,.09,.34,.14,1.05,.14,2.14v16.71c4.07-6.04,7.99-10.76,11.77-14.16,3.78-3.4,7.71-5.09,11.78-5.09,3.24,0,5.88,1.04,7.91,3.11,2.03,2.07,3.05,4.67,3.05,7.79,0,2.79-.78,5.13-2.33,7.03-1.55,1.9-3.5,2.85-5.85,2.85s-4.83-1.19-7.27-3.57c-2.44-2.38-4.36-3.56-5.75-3.56-2.22,0-4.95,1.92-8.18,5.75-3.23,3.84-4.85,7.79-4.85,11.87v45.94c0,5.84,1.27,9.9,3.8,12.17,2.53,2.28,6.75,3.35,12.65,3.21v3.36h-46.88v-3.36Z"/>
    <path class="cls-1" d="M1243.38,359.99c6.72,6.62,10.08,16.01,10.08,28.17h-60.38c.63,15.72,3.96,27.17,9.98,34.35,6.02,7.18,13.15,10.77,21.4,10.77,6.66,0,12.27-1.86,16.83-5.59,4.56-3.73,8.78-9.01,12.65-15.86l3.33,1.22c-2.6,8.64-7.46,16.63-14.6,23.98-7.13,7.35-15.87,11.02-26.2,11.02-11.92,0-21.13-4.82-27.62-14.46-6.5-9.64-9.75-20.74-9.75-33.31,0-13.65,3.77-25.43,11.32-35.35,7.54-9.91,17.37-14.87,29.48-14.87,8.94,0,16.77,3.31,23.49,9.93Zm-45.55,7.79c-2.03,3.67-3.49,8.29-4.37,13.85h40.13c-.7-6.79-1.9-11.85-3.61-15.18-3.11-5.98-8.31-8.96-15.59-8.96s-12.74,3.43-16.54,10.29Z"/>
    <path class="cls-1" d="M1282.65,400.68c5.83-4.07,17.46-9.61,34.9-16.6v-8.66c0-6.93-.63-11.75-1.9-14.46-2.16-4.55-6.62-6.82-13.41-6.82-3.23,0-6.31,.88-9.22,2.65-2.92,1.83-4.37,4.35-4.37,7.54,0,.81,.16,2.19,.48,4.13,.32,1.94,.48,3.18,.48,3.72,0,3.8-1.17,6.45-3.52,7.95-1.33,.88-2.92,1.32-4.75,1.32-2.85,0-5.04-1-6.56-3-1.52-2-2.28-4.23-2.28-6.67,0-4.75,2.74-9.73,8.22-14.92,5.48-5.2,13.52-7.79,24.11-7.79,12.3,0,20.63,4.28,25.01,12.83,2.34,4.69,3.52,11.51,3.52,20.47v40.85c0,3.94,.25,6.66,.76,8.15,.82,2.65,2.54,3.97,5.13,3.97,1.46,0,2.66-.24,3.61-.71,.95-.47,2.6-1.63,4.94-3.46v5.3c-2.03,2.65-4.22,4.82-6.56,6.52-3.55,2.58-7.16,3.87-10.84,3.87-4.31,0-7.43-1.49-9.37-4.48-1.93-2.99-3-6.55-3.19-10.7-4.82,4.48-8.94,7.81-12.36,9.98-5.77,3.67-11.25,5.5-16.45,5.5-5.45,0-10.17-2.05-14.17-6.16-3.99-4.11-5.99-9.32-5.99-15.64,0-9.85,4.6-18.06,13.79-24.65Zm34.9-11.51c-7.29,2.58-13.31,5.43-18.07,8.56-9.13,6.04-13.69,12.9-13.69,20.58,0,6.18,1.9,10.73,5.71,13.65,2.47,1.9,5.23,2.85,8.27,2.85,4.18,0,8.19-1.26,12.03-3.77,3.83-2.51,5.75-5.7,5.75-9.58v-32.29Z"/>
    <path class="cls-1" d="M1351.3,442.95c4.94-.47,8.24-1.36,9.89-2.65,2.53-1.97,3.8-5.91,3.8-11.82v-52.66c0-5.02-.62-8.32-1.85-9.88-1.24-1.56-3.28-2.34-6.13-2.34-1.33,0-2.33,.07-3,.2-.67,.14-1.44,.37-2.33,.71v-3.57l6.85-2.44c2.47-.88,6.53-2.48,12.17-4.79,5.64-2.31,8.62-3.46,8.94-3.46s.51,.17,.57,.51c.06,.34,.09,.99,.09,1.94v13.75c6.28-6.11,11.7-10.34,16.26-12.68,4.56-2.34,9.25-3.51,14.07-3.51,6.53,0,11.73,2.38,15.59,7.13,2.03,2.58,3.71,6.08,5.04,10.49,4.69-5.09,8.78-8.86,12.27-11.31,6.02-4.21,12.17-6.32,18.45-6.32,10.21,0,17.02,4.45,20.44,13.34,1.96,5.03,2.95,12.97,2.95,23.84v42.27c0,4.82,1,8.1,3,9.83,2,1.73,5.59,2.87,10.79,3.41v2.65h-42.88v-2.85c5.52-.54,9.14-1.73,10.89-3.57,1.74-1.83,2.62-5.57,2.62-11.2v-43.9c0-6.59-.67-11.44-2-14.57-2.35-5.57-6.94-8.35-13.79-8.35-4.12,0-8.21,1.46-12.27,4.38-2.35,1.7-5.23,4.42-8.65,8.15v52.15c0,5.5,.9,9.68,2.71,12.53,1.81,2.85,5.62,4.38,11.46,4.58v2.65h-43.65v-2.65c6.02-.81,9.86-2.38,11.51-4.69,1.65-2.31,2.47-8.01,2.47-17.11v-28.46c0-10.44-.63-17.62-1.9-21.55-2.09-6.65-6.53-9.98-13.31-9.98-3.87,0-7.67,1.14-11.41,3.41-3.74,2.28-7.04,5.25-9.89,8.91v55.72c0,5.16,.84,8.73,2.52,10.7,1.68,1.97,5.34,2.99,10.98,3.06v2.65h-43.27v-2.65Z"/>
    <path class="cls-1" d="M1501.35,353.84v-15.45l105.55,51.64v5.6l-105.55,51.64v-15.45l80.28-39.17-80.28-38.83Z"/>
    <path class="cls-1" d="M3.8,693.09c6.78-.68,11.06-1.92,12.84-3.72,1.77-1.8,2.66-6.67,2.66-14.62v-48.59c0-4.41-.29-7.47-.86-9.17-.95-2.78-2.95-4.18-5.99-4.18-.7,0-1.38,.07-2.04,.2-.67,.14-2.58,.68-5.75,1.63v-3.16l4.09-1.43c11.09-3.87,18.83-6.79,23.2-8.76,1.77-.81,2.92-1.22,3.42-1.22,.13,.48,.19,.98,.19,1.53v73.14c0,7.74,.87,12.6,2.62,14.57,1.74,1.97,5.69,3.23,11.84,3.77v2.85H3.8v-2.85Zm15.12-136.21c1.9-2.09,4.25-3.14,7.04-3.14s5.05,1.03,6.99,3.09c1.93,2.06,2.9,4.58,2.9,7.55s-.97,5.39-2.9,7.45c-1.93,2.06-4.26,3.09-6.99,3.09s-5.13-1.03-7.04-3.09c-1.9-2.06-2.85-4.55-2.85-7.45s.95-5.41,2.85-7.5Z"/>
    <path class="cls-1" d="M57.62,693.09c4.5-.61,7.65-1.82,9.46-3.62,1.81-1.8,2.71-5.82,2.71-12.07v-51.75c0-4.31-.38-7.34-1.14-9.09-1.2-2.56-3.68-3.84-7.42-3.84-.57,0-1.16,.04-1.76,.1-.6,.07-1.35,.17-2.23,.31v-3.57c2.6-.81,8.75-2.92,18.45-6.32l8.94-3.16c.44,0,.71,.17,.81,.51,.1,.34,.14,.81,.14,1.43v14.97c5.82-5.84,10.38-9.85,13.67-12.02,4.94-3.33,10.06-4.99,15.38-4.99,4.3,0,8.23,1.32,11.77,3.97,6.83,5.16,10.25,14.36,10.25,27.6v47.57c0,4.89,.91,8.42,2.74,10.59,1.82,2.17,4.88,3.29,9.15,3.36v2.85h-40.6v-2.85c4.63-.68,7.86-2.05,9.7-4.13,1.84-2.07,2.76-6.57,2.76-13.5v-43.5c0-5.84-1.01-10.68-3.03-14.52-2.02-3.84-5.74-5.76-11.16-5.76-3.72,0-7.5,1.36-11.35,4.07-2.14,1.56-4.92,4.14-8.32,7.74v57.04c0,4.89,1.01,8.17,3.04,9.83,2.03,1.67,5.23,2.56,9.6,2.7v2.85H57.62v-2.85Z"/>
    <path class="cls-1" d="M201.02,602.02v7.33h-19.4l-.19,58.67c0,5.16,.41,9.07,1.24,11.71,1.52,4.69,4.5,7.03,8.94,7.03,2.28,0,4.26-.58,5.94-1.73,1.68-1.15,3.6-2.99,5.75-5.5l2.47,2.24-2.09,3.06c-3.3,4.75-6.78,8.12-10.46,10.08-3.68,1.97-7.23,2.95-10.65,2.95-7.48,0-12.55-3.57-15.21-10.7-1.46-3.87-2.19-9.23-2.19-16.09v-61.73h-10.36c-.32-.2-.56-.41-.71-.61-.16-.2-.24-.47-.24-.81,0-.68,.14-1.2,.43-1.58,.29-.37,1.19-1.24,2.71-2.6,4.37-3.87,7.53-7.01,9.46-9.42,1.93-2.41,6.48-8.78,13.65-19.1,.82,0,1.31,.07,1.47,.2,.16,.14,.24,.65,.24,1.53v25.06h19.21Z"/>
    <path class="cls-1" d="M257.4,693.29c4.94-.47,8.24-1.36,9.89-2.65,2.53-1.97,3.8-5.91,3.8-11.82v-52.66c0-5.02-.62-8.32-1.85-9.88-1.24-1.56-3.28-2.34-6.13-2.34-1.33,0-2.33,.07-3,.2-.67,.14-1.44,.37-2.33,.71v-3.57l6.85-2.44c2.47-.88,6.53-2.48,12.17-4.79,5.64-2.31,8.62-3.46,8.94-3.46s.51,.17,.57,.51c.06,.34,.09,.99,.09,1.94v13.75c6.28-6.11,11.7-10.34,16.26-12.68,4.56-2.34,9.25-3.51,14.07-3.51,6.53,0,11.73,2.38,15.59,7.13,2.03,2.58,3.71,6.08,5.04,10.49,4.69-5.09,8.78-8.86,12.27-11.31,6.02-4.21,12.17-6.32,18.45-6.32,10.21,0,17.02,4.45,20.44,13.34,1.96,5.03,2.95,12.97,2.95,23.84v42.27c0,4.82,1,8.1,3,9.83,2,1.73,5.59,2.87,10.79,3.41v2.65h-42.88v-2.85c5.52-.54,9.14-1.73,10.89-3.57,1.74-1.83,2.62-5.57,2.62-11.2v-43.9c0-6.59-.67-11.44-2-14.57-2.35-5.57-6.94-8.35-13.79-8.35-4.12,0-8.21,1.46-12.27,4.38-2.35,1.7-5.23,4.42-8.65,8.15v52.15c0,5.5,.9,9.68,2.71,12.53,1.81,2.85,5.62,4.38,11.46,4.58v2.65h-43.65v-2.65c6.02-.81,9.86-2.38,11.51-4.69,1.65-2.31,2.47-8.01,2.47-17.11v-28.46c0-10.44-.63-17.62-1.9-21.55-2.09-6.65-6.53-9.98-13.31-9.98-3.87,0-7.67,1.14-11.41,3.41-3.74,2.28-7.04,5.25-9.89,8.91v55.72c0,5.16,.84,8.73,2.52,10.7,1.68,1.97,5.34,2.99,10.98,3.06v2.65h-43.27v-2.65Z"/>
    <path class="cls-1" d="M426.66,651.02c5.83-4.07,17.46-9.61,34.9-16.6v-8.66c0-6.93-.63-11.75-1.9-14.46-2.16-4.55-6.62-6.82-13.41-6.82-3.23,0-6.31,.88-9.22,2.65-2.92,1.83-4.37,4.35-4.37,7.54,0,.81,.16,2.19,.48,4.13,.32,1.94,.48,3.18,.48,3.72,0,3.8-1.17,6.45-3.52,7.95-1.33,.88-2.92,1.32-4.75,1.32-2.85,0-5.04-1-6.56-3-1.52-2-2.28-4.23-2.28-6.67,0-4.75,2.74-9.73,8.23-14.92,5.48-5.2,13.52-7.79,24.1-7.79,12.3,0,20.63,4.28,25.01,12.83,2.34,4.69,3.52,11.51,3.52,20.47v40.85c0,3.94,.25,6.66,.76,8.15,.82,2.65,2.53,3.97,5.13,3.97,1.46,0,2.66-.24,3.61-.71,.95-.47,2.6-1.63,4.94-3.46v5.3c-2.03,2.65-4.22,4.82-6.56,6.52-3.55,2.58-7.16,3.87-10.84,3.87-4.31,0-7.43-1.49-9.37-4.48-1.93-2.99-3-6.55-3.19-10.7-4.82,4.48-8.94,7.81-12.36,9.98-5.77,3.67-11.25,5.5-16.45,5.5-5.45,0-10.17-2.05-14.17-6.16-3.99-4.11-5.99-9.32-5.99-15.64,0-9.85,4.6-18.06,13.79-24.65Zm34.9-11.51c-7.29,2.58-13.31,5.43-18.07,8.56-9.13,6.04-13.69,12.9-13.69,20.58,0,6.18,1.9,10.73,5.71,13.65,2.47,1.9,5.23,2.85,8.27,2.85,4.18,0,8.19-1.26,12.03-3.77,3.83-2.51,5.75-5.7,5.75-9.58v-32.29Z"/>
    <path class="cls-1" d="M495.98,693.09c6.78-.68,11.06-1.92,12.84-3.72,1.77-1.8,2.66-6.67,2.66-14.62v-48.59c0-4.41-.29-7.47-.86-9.17-.95-2.78-2.95-4.18-5.99-4.18-.7,0-1.38,.07-2.04,.2-.67,.14-2.58,.68-5.75,1.63v-3.16l4.09-1.43c11.09-3.87,18.83-6.79,23.2-8.76,1.77-.81,2.92-1.22,3.42-1.22,.13,.48,.19,.98,.19,1.53v73.14c0,7.74,.87,12.6,2.62,14.57,1.74,1.97,5.69,3.23,11.84,3.77v2.85h-46.21v-2.85Zm15.12-136.21c1.9-2.09,4.25-3.14,7.04-3.14s5.05,1.03,6.99,3.09c1.93,2.06,2.9,4.58,2.9,7.55s-.97,5.39-2.9,7.45c-1.93,2.06-4.26,3.09-6.99,3.09s-5.13-1.03-7.04-3.09c-1.9-2.06-2.85-4.55-2.85-7.45s.95-5.41,2.85-7.5Z"/>
    <path class="cls-1" d="M549.8,693.09c4.5-.61,7.65-1.82,9.46-3.62,1.81-1.8,2.71-5.82,2.71-12.07v-51.75c0-4.31-.38-7.34-1.14-9.09-1.2-2.56-3.68-3.84-7.42-3.84-.57,0-1.16,.04-1.76,.1-.6,.07-1.35,.17-2.23,.31v-3.57c2.6-.81,8.75-2.92,18.45-6.32l8.94-3.16c.44,0,.71,.17,.81,.51,.1,.34,.14,.81,.14,1.43v14.97c5.82-5.84,10.38-9.85,13.67-12.02,4.94-3.33,10.06-4.99,15.38-4.99,4.3,0,8.23,1.32,11.77,3.97,6.83,5.16,10.25,14.36,10.25,27.6v47.57c0,4.89,.91,8.42,2.74,10.59,1.82,2.17,4.88,3.29,9.15,3.36v2.85h-40.6v-2.85c4.63-.68,7.86-2.05,9.7-4.13,1.84-2.07,2.76-6.57,2.76-13.5v-43.5c0-5.84-1.01-10.68-3.03-14.52-2.02-3.84-5.74-5.76-11.16-5.76-3.72,0-7.5,1.36-11.35,4.07-2.14,1.56-4.92,4.14-8.32,7.74v57.04c0,4.89,1.01,8.17,3.04,9.83,2.03,1.67,5.23,2.56,9.6,2.7v2.85h-41.55v-2.85Z"/>
    <path class="cls-1" d="M672.37,581.34c8.11-10.93,17.69-19.73,28.72-26.38l1.81,3.36c-10.21,8.69-17.5,17.15-21.87,25.36-7.67,14.33-11.51,33.41-11.51,57.25,0,17.66,1.46,32.22,4.37,43.7,5.13,20.1,14.8,35.14,29,45.12l-2.47,3.36c-7.8-4.01-16.23-11.85-25.29-23.53-15.02-19.35-22.54-41.36-22.54-66.01s6.59-44.38,19.78-62.24Z"/>
    <path class="cls-1" d="M760.23,673.33c-2.47,9.03-6.09,17.56-10.84,25.57-5.01,8.56-11.7,16.54-20.06,23.94-5.52,4.89-10.14,8.32-13.88,10.29l-1.81-3.36c9.76-7.95,16.89-16.06,21.4-24.35,7.99-14.67,11.98-34.06,11.98-58.16,0-19.49-1.84-35.41-5.52-47.77-5.13-17.45-14.42-31.17-27.86-41.15l2.47-3.36c10.02,5.98,19.33,15.01,27.96,27.1,13.25,18.61,19.87,39.46,19.87,62.54,0,10.12-1.24,19.69-3.71,28.73Z"/>
    <path class="cls-1" d="M834.21,640.32c7.67-.88,13.53-2.72,17.59-5.51,6.21-4.35,9.32-11.29,9.32-20.81l.19-28.05c0-12.58,4.58-21.14,13.76-25.7,5.31-2.65,14.74-4.42,28.27-5.3v4.58c-6.85,.82-13.11,3.21-18.78,7.19-5.67,3.98-8.51,10.52-8.51,19.63v24.99c0,10.74-2.4,18.12-7.21,22.13-4.8,4.01-13.34,7.34-25.61,9.99l-3.32,.71v.51c12.96,1.09,22.19,4.24,27.69,9.46,5.5,5.22,8.31,12.11,8.44,20.65l.29,28.69c.06,8.41,3.26,14.72,9.6,18.92,3.61,2.37,9.41,4.41,17.4,6.1v4.58c-12.71,0-22.9-2.34-30.55-7.02-7.65-4.68-11.48-12.55-11.48-23.61l-.19-31.34c0-7.87-3.3-13.8-9.89-17.8-3.74-2.24-9.41-4.1-17.02-5.6v-7.44Z"/>
    <path class="cls-1" d="M195.6,942.92c5.83-.54,9.7-1.65,11.6-3.31,1.9-1.66,2.85-5.21,2.85-10.64v-45.94c0-6.72-.59-11.53-1.76-14.41-1.17-2.89-3.31-4.33-6.42-4.33-.63,0-1.47,.09-2.52,.25-1.05,.17-2.14,.39-3.28,.66v-3.36c3.6-1.36,7.3-2.75,11.09-4.18,3.79-1.43,6.41-2.44,7.87-3.06,3.16-1.29,6.41-2.75,9.76-4.38,.44,0,.71,.17,.81,.51,.1,.34,.14,1.05,.14,2.14v16.71c4.07-6.04,7.99-10.76,11.77-14.16,3.78-3.39,7.71-5.09,11.78-5.09,3.24,0,5.88,1.04,7.91,3.11,2.03,2.07,3.05,4.67,3.05,7.79,0,2.79-.78,5.13-2.33,7.03-1.55,1.9-3.5,2.85-5.85,2.85s-4.83-1.19-7.27-3.57c-2.44-2.38-4.36-3.57-5.75-3.57-2.22,0-4.94,1.92-8.18,5.76-3.23,3.84-4.85,7.79-4.85,11.87v45.94c0,5.84,1.27,9.9,3.8,12.17,2.54,2.28,6.75,3.35,12.65,3.21v3.36h-46.88v-3.36Z"/>
    <path class="cls-1" d="M327.67,860.66c6.72,6.62,10.08,16.01,10.08,28.17h-60.38c.63,15.72,3.96,27.17,9.98,34.35,6.02,7.18,13.15,10.77,21.4,10.77,6.66,0,12.27-1.86,16.83-5.59,4.56-3.73,8.78-9.01,12.65-15.86l3.33,1.22c-2.6,8.64-7.46,16.63-14.6,23.98-7.13,7.35-15.86,11.02-26.2,11.02-11.92,0-21.13-4.82-27.62-14.46-6.5-9.64-9.75-20.75-9.75-33.31,0-13.65,3.77-25.43,11.32-35.35,7.54-9.91,17.37-14.87,29.48-14.87,8.94,0,16.77,3.31,23.49,9.93Zm-45.55,7.79c-2.03,3.67-3.49,8.29-4.37,13.85h40.13c-.7-6.79-1.9-11.85-3.61-15.18-3.11-5.97-8.31-8.96-15.59-8.96s-12.74,3.43-16.55,10.29Z"/>
    <path class="cls-1" d="M395.57,852.36v7.33h-19.4l-.19,58.67c0,5.16,.41,9.07,1.24,11.71,1.52,4.69,4.5,7.03,8.94,7.03,2.28,0,4.26-.58,5.94-1.73,1.68-1.15,3.6-2.99,5.75-5.5l2.47,2.24-2.09,3.06c-3.3,4.75-6.78,8.12-10.46,10.08-3.68,1.97-7.23,2.95-10.65,2.95-7.48,0-12.55-3.57-15.21-10.7-1.46-3.87-2.19-9.24-2.19-16.09v-61.73h-10.36c-.32-.2-.56-.41-.71-.61-.16-.2-.24-.47-.24-.82,0-.68,.14-1.21,.43-1.58,.29-.37,1.19-1.24,2.71-2.6,4.37-3.87,7.53-7.01,9.46-9.42,1.93-2.41,6.48-8.78,13.65-19.1,.82,0,1.31,.07,1.47,.2,.16,.14,.24,.65,.24,1.53v25.06h19.21Z"/>
    <path class="cls-1" d="M430.28,851.95v65.19c0,4.62,.6,8.32,1.81,11.1,2.34,5.16,6.66,7.74,12.93,7.74,4.31,0,8.53-1.53,12.65-4.58,2.34-1.7,4.72-4.04,7.13-7.03v-55.82c0-5.23-.95-8.66-2.85-10.29s-5.71-2.61-11.41-2.95v-3.36h30.71v71.71c0,4.62,.78,7.79,2.33,9.52,1.55,1.73,4.83,2.5,9.84,2.29v2.85c-3.49,1.02-6.05,1.78-7.7,2.29-1.65,.51-4.41,1.44-8.27,2.8-1.65,.61-5.26,2.07-10.84,4.38-.32,0-.51-.15-.57-.46-.06-.31-.1-.66-.1-1.07v-16.4c-4.31,5.5-8.24,9.58-11.79,12.22-5.39,4.08-11.09,6.11-17.12,6.11-5.52,0-10.71-2.11-15.59-6.32-4.94-4.15-7.42-11.12-7.42-20.91v-52.93c0-5.39-1.08-9-3.23-10.82-1.4-1.14-4.37-1.95-8.94-2.42v-2.85h28.43Z"/>
    <path class="cls-1" d="M498.36,942.92c5.83-.54,9.7-1.65,11.6-3.31,1.9-1.66,2.85-5.21,2.85-10.64v-45.94c0-6.72-.59-11.53-1.76-14.41-1.17-2.89-3.31-4.33-6.42-4.33-.63,0-1.47,.09-2.52,.25-1.05,.17-2.14,.39-3.28,.66v-3.36c3.6-1.36,7.3-2.75,11.09-4.18,3.79-1.43,6.41-2.44,7.87-3.06,3.16-1.29,6.41-2.75,9.76-4.38,.44,0,.71,.17,.81,.51,.1,.34,.14,1.05,.14,2.14v16.71c4.07-6.04,7.99-10.76,11.77-14.16,3.78-3.39,7.71-5.09,11.78-5.09,3.24,0,5.88,1.04,7.91,3.11,2.03,2.07,3.05,4.67,3.05,7.79,0,2.79-.78,5.13-2.33,7.03-1.55,1.9-3.5,2.85-5.85,2.85s-4.83-1.19-7.27-3.57c-2.44-2.38-4.36-3.57-5.75-3.57-2.22,0-4.94,1.92-8.18,5.76-3.23,3.84-4.85,7.79-4.85,11.87v45.94c0,5.84,1.27,9.9,3.8,12.17,2.54,2.28,6.75,3.35,12.65,3.21v3.36h-46.88v-3.36Z"/>
    <path class="cls-1" d="M565.87,943.43c4.5-.61,7.65-1.82,9.46-3.62,1.81-1.8,2.71-5.82,2.71-12.07v-51.75c0-4.31-.38-7.34-1.14-9.09-1.2-2.56-3.68-3.84-7.42-3.84-.57,0-1.16,.03-1.76,.1-.6,.07-1.35,.17-2.23,.31v-3.57c2.6-.81,8.75-2.92,18.45-6.32l8.94-3.16c.44,0,.71,.17,.81,.51,.1,.34,.14,.81,.14,1.43v14.97c5.82-5.84,10.38-9.85,13.67-12.02,4.94-3.33,10.06-4.99,15.38-4.99,4.3,0,8.23,1.32,11.77,3.97,6.83,5.16,10.25,14.36,10.25,27.6v47.57c0,4.89,.91,8.42,2.74,10.59,1.82,2.17,4.88,3.29,9.15,3.36v2.85h-40.6v-2.85c4.63-.68,7.86-2.05,9.7-4.12,1.84-2.07,2.76-6.57,2.76-13.5v-43.5c0-5.84-1.01-10.68-3.03-14.52-2.02-3.84-5.74-5.76-11.16-5.76-3.72,0-7.5,1.36-11.35,4.08-2.14,1.56-4.92,4.14-8.32,7.74v57.04c0,4.89,1.01,8.17,3.04,9.83,2.03,1.67,5.23,2.56,9.6,2.7v2.85h-41.55v-2.85Z"/>
    <path class="cls-1" d="M788.66,830.67c6.4,13.24,9.6,28.45,9.6,45.63,0,13.51-1.97,26.25-5.9,38.2-7.42,22.48-20.25,33.72-38.51,33.72-12.49,0-22.76-6.08-30.81-18.23-8.62-12.97-12.93-30.59-12.93-52.87,0-17.52,2.88-32.6,8.65-45.23,7.8-17.18,19.78-25.77,35.94-25.77,14.58,0,25.9,8.18,33.95,24.55Zm-14.45,95.36c3.49-10.92,5.23-26.73,5.23-47.43,0-16.49-1.05-29.65-3.14-39.49-3.93-18.32-11.28-27.48-22.06-27.48s-18.16,9.43-22.16,28.3c-2.09,10.04-3.14,23.28-3.14,39.7,0,15.4,1.08,27.72,3.23,36.95,4.06,17.24,11.66,25.85,22.82,25.85,9.32,0,15.72-5.46,19.21-16.39Z"/>
    <path class="cls-1" d="M842.39,963.14c-4.56,5.94-9.57,10.24-15.02,12.89l-1.81-4.07c6.09-3.67,10.28-7.64,12.6-11.92,2.31-4.28,3.47-7.84,3.47-10.7,0-.95-.29-1.56-.86-1.83-.57-.27-1.17-.41-1.81-.41l-6.37,.71c-2.66,0-5.1-.93-7.32-2.8-2.22-1.87-3.33-4.57-3.33-8.1,0-2.72,.92-5.36,2.76-7.95,1.84-2.58,4.85-3.87,9.03-3.87s7.57,1.68,10.75,5.04c3.17,3.36,4.75,8.03,4.75,14.01,0,6.72-2.28,13.05-6.85,19Zm-17.07-92.9c-2.06-2.27-3.09-5.01-3.09-8.2s1.05-5.92,3.14-8.2c2.09-2.27,4.63-3.41,7.61-3.41s5.53,1.14,7.65,3.41c2.12,2.28,3.19,5.01,3.19,8.2s-1.05,5.93-3.14,8.2c-2.09,2.28-4.66,3.41-7.7,3.41s-5.6-1.14-7.65-3.41Z"/>
    <path class="cls-1" d="M81.3,1148.44c-7.8,1.56-13.54,3.46-17.21,5.7-6.4,4-9.6,9.9-9.6,17.7l-.19,31.34c0,10.99-3.8,18.84-11.41,23.55-7.61,4.71-17.88,7.07-30.81,7.07v-4.58c7.99-1.7,13.79-3.73,17.4-6.1,6.47-4.21,9.7-10.51,9.7-18.92l.19-28.69c0-9.23,3.07-16.28,9.22-21.16,6.15-4.88,15.18-7.87,27.1-8.95v-.51c-13.83-2.92-22.78-5.85-26.86-8.77-6.44-4.62-9.65-12.65-9.65-24.07v-24.99c0-9.72-3.75-16.93-11.24-21.62-4.22-2.65-9.51-4.38-15.86-5.2v-4.58c16.29,0,27.4,2.81,33.33,8.42,5.93,5.61,8.89,13.14,8.89,22.59l.19,28.05c0,9.86,3.33,16.93,9.98,21.21,3.8,2.52,9.41,4.22,16.83,5.1v7.44Z"/>
  </g>
</svg>
"""


def svg_sample_silhouette_points(svg_string, len_per_point) -> list[list[float]]:
    points = []
    svg_dom = minidom.parseString(svg_string)
    for path_elem in tqdm(svg_dom.getElementsByTagName("path")):
        remain_len = 0.0
        for curve in parse_path(path_elem.getAttribute("d")):
            if isinstance(curve, svg.path.path.Close):
                remain_len = 0.0
                continue
            if (curve_len := curve.length()) <= remain_len:
                remain_len -= curve_len
                continue
            for step in (steps := np.arange(start=remain_len / curve_len, stop=1.0, step=len_per_point / curve_len)):
                point = curve.point(step)   # step ∈ [0, 1)
                points.append([point.real, point.imag])
            remain_len = len_per_point - (1.0 - steps[-1]) * curve_len
    return points


LEN_PER_POINTS = 1
sil_points = svg_sample_silhouette_points(SVG_STRING, LEN_PER_POINTS)

x, y = zip(*sil_points)
plt.plot(x, y, '.', markersize=1)
plt.xlim(0, 2284)
plt.ylim(1306, 0)

