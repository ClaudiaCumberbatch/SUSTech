package exer;

public class Car {
    String carNumber;
    Calculate calculate;

    public Car(String carNumber) {
        this.carNumber = carNumber;
    }

    public void setCalculate(Calculate calculate) {
        this.calculate = calculate;
    }

    public String getCarNumber() {
        return carNumber;
    }

    public int parkingPrice(double hours) {
        return this.calculate.calculateParkingPrice(hours);
    }

    public int increasePoints(double hours) {
        return this.calculate.calculatePoints(hours);
    }
}