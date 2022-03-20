package kapablankaNew.JeuroNet.Recurrent;

/*
This class implements different types of recurrent neural networks topologies
In all the following schemes xi - input values, yi - output values, hi - recurrent cell

ALL_INPUT_ALL_OUTPUT:

    yo    y1    y2    y3    y4
    ^     ^     ^     ^     ^
    |     |     |     |     |
    h0 -> h1 -> h2 -> h3 -> h4
    ^     ^     ^     ^     ^
    |     |     |     |     |
    x0    x1    x2    x3    x4

NO_INPUT:

    yo    y1    y2    y3    y4
    ^     ^     ^     ^     ^
    |     |     |     |     |
    h0 -> h1 -> h2 -> h3 -> h4
    ^     ^     ^
    |     |     |
    x0    x1    x2

NO_OUTPUT:

                y2    y3    y4
                ^     ^     ^
                |     |     |
    h0 -> h1 -> h2 -> h3 -> h4
    ^     ^     ^     ^     ^
    |     |     |     |     |
    x0    x1    x2    x3    x4

NO_INPUT_NO_OUTPUT:

                y2    y3    y4
                ^     ^     ^
                |     |     |
    h0 -> h1 -> h2 -> h3 -> h4
    ^     ^     ^
    |     |     |
    x0    x1    x2
*/

public enum RecurrentLayerType {
    ALL_INPUT_ALL_OUTPUT,
    NO_INPUT,
    NO_OUTPUT,
    NO_INPUT_NO_OUTPUT
}
