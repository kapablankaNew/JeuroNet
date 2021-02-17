package org.kapablankaNew.JeuroNet;

import lombok.NonNull;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

public class PictureConverter {
    private final int newWidth;
    private final int newHeight;
    private int boundary = 128;
    private int type;
    private int width;
    private int height;
    private int groupWidth;
    private int groupHeight;
    private boolean isGrayScale;

    public PictureConverter(int width, int height) {
        //input parameters width and height - this is size of converting image
        //fields parameters width and height - size of raw image
        this.width = 0;
        this.height = 0;
        groupWidth = 1;
        groupHeight = 1;
        newWidth = width;
        newHeight = height;
        //if true - image convert to gray scale, if else - to black-white
        isGrayScale = true;
    }

    public PictureConverter(int width, int height, int boundary) {
        //if comes value of boundary - using gray scale
        this(width, height);
        isGrayScale = false;
        this.boundary = boundary;
    }

    public List<Double> convertImageToListOfSignals(@NonNull BufferedImage image) {
        List<Double> result = new ArrayList<>();
        //updating data from raw image
        width = image.getWidth();
        height = image.getHeight();
        type = image.getType();
        BufferedImage newImage = image;
        //if image less than final size - using zoom
        if (width < newWidth || height < newHeight) {
            newImage = zoom(newImage);
            width = newImage.getWidth();
            height = newImage.getHeight();
        }
        // if width greater thar final - reduce
        if (width > newWidth) {
            newImage = reduceHorizontally(newImage);
            width = newImage.getWidth();
        }
        //also with height
        if (height > newHeight) {
            newImage = reduceVertically(newImage);
            height = newImage.getHeight();
        }
        //of this step raw image have required size
        //because of this, convert several pixel to double value
        //and adding to result
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                Color color = new Color(newImage.getRGB(i, j));
                result.add(Brightness(color));
            }
        }
        return result;
    }

    private BufferedImage reduceHorizontally(@NonNull BufferedImage image) {
        BufferedImage newImage = new BufferedImage(newWidth, height, image.getType());
        //converting group pixel to one pixel
        groupHeight = 1;
        int step = image.getWidth() % newWidth;
        for (int y = 0; y < height; y++) {
            //calculate width of group
            groupWidth = image.getWidth() / newWidth;
            int counter = 0;
            for (int x = 0; x < width; x += groupWidth) {
                //if width%groupWidth != 0 - in one moment changing the size of group
                if (counter == newWidth - step) {
                    groupWidth++;
                }
                //calculate average value of pixels in group
                Color color = getAverageColor(x, y, image);
                //set pixel in new image as average value of group
                newImage.setRGB(counter, y, color.getRGB());
                counter++;
            }
        }
        return newImage;
    }

    private BufferedImage reduceVertically(@NonNull BufferedImage image) {
        BufferedImage newImage = new BufferedImage(width, newHeight, image.getType());
        //converting group pixel to one pixel
        groupWidth = 1;
        int step = image.getHeight() % newHeight;
        for (int x = 0; x < width; x++) {
            //calculate height of group
            groupHeight = image.getHeight() / newHeight;
            int counter = 0;
            for (int y = 0; y < height; y += groupHeight) {
                //if height%groupHeight != 0 - in one moment changing the size of group
                if (counter == newHeight - step) {
                    groupHeight++;
                }
                //calculate average value of pixels in group
                Color color = getAverageColor(x, y, image);
                //set pixel in new image as average value of group
                newImage.setRGB(x, counter, color.getRGB());
                counter++;
            }
        }
        return newImage;
    }

    private BufferedImage zoom(@NonNull BufferedImage image) {
        //increasing the size of the image without loss of quality
        //method - bicubic interpolation
        BufferedImage newImage = new BufferedImage(Math.max(width, newWidth),
                Math.max(height, newHeight), image.getType());
        Graphics2D graphics2D = newImage.createGraphics();
        graphics2D.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        graphics2D.drawImage(image, 0, 0, newImage.getWidth(), newImage.getHeight(),
                null);
        return newImage;
    }

    //method for converting group of pixel to one pixel with average value of color
    private Color getAverageColor(int x, int y, BufferedImage image) {
        int sumR = 0;
        int sumG = 0;
        int sumB = 0;
        int ch = 0;
        //calculate average values of R, G and B
        for (int i = x; i < x + groupWidth && i < width; i++) {
            for (int j = y; j < y + groupHeight && j < height; j++) {
                Color color = new Color(image.getRGB(i, j));
                sumR += color.getRed();
                sumG += color.getGreen();
                sumB += color.getBlue();
                ch++;
            }
        }
        int R = sumR / ch;
        int G = sumG / ch;
        int B = sumB / ch;
        //return new pixel with average values of colors
        return new Color(R, G, B);
    }

    //method for convert color pixel to black-white
    private double Brightness(Color color) {
        int R = color.getRed();
        int G = color.getGreen();
        int B = color.getBlue();
        //this is formula from internet))
        final double brightness = 0.299 * R + 0.587 * G + 0.114 * B;
        //return pixel in gray scale, or white or black
        if (isGrayScale) {
            return brightness;
        }
        return brightness > boundary ? 255 : 0;
    }

    public BufferedImage getImageFromSignals(List<Double> pixels) {
        //create new image
        BufferedImage image = new BufferedImage(width, height, type);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                //converting each number to pixel
                int R, G, B;
                if (isGrayScale) {
                    R = pixels.get(i * height + j).intValue();
                    G = pixels.get(i * height + j).intValue();
                    B = pixels.get(i * height + j).intValue();
                } else {
                    R = pixels.get(i * height + j).intValue() > boundary ? 255 : 0;
                    G = pixels.get(i * height + j).intValue() > boundary ? 255 : 0;
                    B = pixels.get(i * height + j).intValue() > boundary ? 255 : 0;
                }
                Color color = new Color(R, G, B);
                image.setRGB(i, j, color.getRGB());
            }
        }
        return image;
    }
}