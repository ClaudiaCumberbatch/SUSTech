public class AirConditioner {
    private String location;
    private boolean airConditionerOn;

    public AirConditioner(String location) {
        this.location = location;
        airConditionerOn = false;
    }

    public void on() {
        airConditionerOn = true;
        System.out.println(this);
    }

    public void off() {
        airConditionerOn = false;
        System.out.println(this);
    }

    public String toString() {
        return String.format("%s: Air Conditioner is %s", location, airConditionerOn ? "On" : "Off");
    }

}
